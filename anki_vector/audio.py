# Copyright (c) 2019 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Support for accessing Vector's audio.

Vector's speakers can be used for playing user-provided audio.
TODO Ability to access the Vector's audio stream to come.

The :class:`AudioComponent` class defined in this module is made available as
:attr:`anki_vector.robot.Robot.audio` and can be used to play audio data on the robot.
"""

# __all__ should order by constants, event classes, other classes, functions.
__all__ = ['AudioComponent']

import asyncio
from concurrent import futures
from email.mime import audio
from enum import Enum
import time
import wave
import io
import os
from pathlib import Path

import aiohttp
from pydub import AudioSegment
from urllib.parse import urlparse
from google.protobuf.text_format import MessageToString
from . import util
from .connection import on_connection_thread
from .exceptions import VectorExternalAudioPlaybackException
from .messaging import protocol

MAX_ROBOT_AUDIO_CHUNK_SIZE = 1024  # 1024 is maximum, larger sizes will fail
DEFAULT_FRAME_SIZE = MAX_ROBOT_AUDIO_CHUNK_SIZE // 2


class RobotVolumeLevel(Enum):
    """Use these values for setting the master audio volume.  See :meth:`set_master_volume`

    Note that muting the robot is not supported from the SDK.
    """
    LOW = 0
    MEDIUM_LOW = 1
    MEDIUM = 2
    MEDIUM_HIGH = 3
    HIGH = 4


class AudioComponent(util.Component):
    """Handles audio playback on Vector's speaker.

        The `AudioComponent` class is responsible for managing audio playback on Vector. It can play audio from
        both local file paths and URLs, automatically transcode audio to WAV format (16000 Hz, 16 bits, 1 channel),
        and stream it to Vector’s speaker. The class manages the state of audio playback and ensures smooth streaming
        by sending audio in chunks.

        This class is typically owned by an instance of :class:`anki_vector.robot.Robot` or
        :class:`anki_vector.robot.AsyncRobot`.

        Key Features:
        -------------
        - **Local and URL Audio Playback**: Plays audio from both local files and remote URLs.
        - **In-memory Transcoding**: Automatically transcodes audio to the required WAV format.
        - **Real-time Audio Streaming**: Streams audio data in real-time to Vector’s speaker in chunks, handling timing to ensure correct playback.
        - **Volume Control**: Allows control over the audio playback volume (0-100).
        - **Concurrency Handling**: Ensures no other audio is played concurrently, raising exceptions if a playback conflict arises.

        Example usage with a local file:

        .. testcode::

            import anki_vector

            with anki_vector.Robot() as robot:
                robot.audio.stream_audio('../examples/sounds/vector_alert.wav')

        Example usage with a URL:

        .. testcode::

            import anki_vector

            with anki_vector.Robot() as robot:
                robot.audio.stream_audio('https://example.com/sounds/vector_alert.wav')

        Key Methods:
        ------------
        - **stream_audio(source, volume=50)**: Plays audio from a local file path or URL, with optional volume control.
        - **_load_audio(source)**: Loads the audio into memory from a local path or URL.
        - **_transcode_audio(audio)**: Transcodes audio to WAV format and stores it in a `BytesIO` buffer.
        - **_play_audio(audio_buffer, volume)**: Streams the audio buffer to Vector's speakers.
        - **_request_handler(reader, params, volume)**: Manages sending audio data to Vector in chunks, handling playback timing and stream completion.

        Raises:
        -------
        - **VectorExternalAudioPlaybackException**: Raised if there is an issue with the audio format, volume range, or if another sound is already playing.

        Notes:
        ------
        - This class requires gRPC interface methods to stream audio to Vector's hardware, and it must be run within an asynchronous context.

    """

    # TODO restore audio feed code when ready

    def __init__(self, robot):
        super().__init__(robot)

        self._is_shutdown = False
        # don't create asyncio.Events here, they are not thread-safe
        self._is_active_event = None
        self._done_event = None
        self._is_streaming = False

    @on_connection_thread(requires_control=False)
    async def set_master_volume(self, volume: RobotVolumeLevel) -> protocol.MasterVolumeResponse:
        """Sets Vector's master volume level.

        Note that muting the robot is not supported from the SDK.

        .. testcode::

            import anki_vector
            from anki_vector import audio

            with anki_vector.Robot(behavior_control_level=None) as robot:
                robot.audio.set_master_volume(audio.RobotVolumeLevel.MEDIUM_HIGH)

        :param volume: the robot's desired volume
        """

        volume_request = protocol.MasterVolumeRequest(volume_level=volume.value)
        return await self.conn.grpc_interface.SetMasterVolume(volume_request)

    def _open_file(self, audio_buffer):
        """Opens a BytesIO buffer containing WAV audio data and checks format compatibility.

            This method reads a `BytesIO` buffer as a WAV file and verifies that the audio
            format meets the required specifications (16000 Hz, 16-bit, mono). If the format
            is not compatible, an exception is raised.

            :param audio_buffer: A `BytesIO` buffer containing the WAV audio data.
            :return: A tuple containing the `wave` reader object and the audio parameters.
            :raises VectorExternalAudioPlaybackException: If the audio format does not meet the required specifications.

            Example usage:

            .. testcode::

                reader, params = robot.audio._open_file(transcoded_audio_buffer)
        """
        # Open the BytesIO buffer as a wav file
        _reader = wave.open(audio_buffer, 'rb')
        _params = _reader.getparams()
        self.logger.info("Playing audio from BytesIO buffer")

        # Ensure that the audio format is compatible
        if _params.sampwidth != 2 or _params.nchannels != 1 or _params.framerate > 16025 or _params.framerate < 8000:
            raise VectorExternalAudioPlaybackException(
                f"Audio format must be 8000-16025 hz, 16 bits, 1 channel.  "
                f"Found {_params.framerate} hz/{_params.sampwidth * 8} bits/{_params.nchannels} channels")

        return _reader, _params

    async def _request_handler(self, reader, params, volume):
        """Handles generating and sending request messages for the AudioPlaybackStream.

            This method streams audio data in chunks to Vector's speakers. It reads audio frames
            from the provided `reader`, prepares the playback request, and sends chunks of the
            audio data to Vector in real-time. It manages the timing of audio playback to ensure
            it does not get ahead of the expected playback time.

            :param reader: A `wave` reader object used to read audio frames from the WAV data.
            :param params: The audio parameters (`wave` params) containing details such as the number of frames,
                           sample width, channels, and framerate.
            :param volume: The audio playback volume (0-100), where 0 is mute and 100 is the maximum volume.

            :yield: Audio playback requests to be sent to Vector.

            :raises VectorExternalAudioPlaybackException: If the audio format or playback parameters are invalid.

            Example usage:

            .. testcode::

                async for msg in robot.audio._request_handler(reader, params, volume=75):
                    await grpc_interface.ExternalAudioStreamPlayback(msg)
        """
        frames = params.nframes  # 16 bit samples, not bytes

        # send preparation message
        msg = protocol.ExternalAudioStreamPrepare(audio_frame_rate=params.framerate, audio_volume=volume)
        msg = protocol.ExternalAudioStreamRequest(audio_stream_prepare=msg)

        yield msg
        await asyncio.sleep(0)  # give event loop a chance to process messages

        # count of full and partial chunks
        total_chunks = (frames + DEFAULT_FRAME_SIZE - 1) // DEFAULT_FRAME_SIZE
        curr_chunk = 0
        start_time = time.time()
        self.logger.debug("Starting stream time %f", start_time)

        while frames > 0 and not self._done_event.is_set():
            read_count = min(frames, DEFAULT_FRAME_SIZE)
            audio_data = reader.readframes(read_count)
            msg = protocol.ExternalAudioStreamChunk(audio_chunk_size_bytes=len(audio_data),
                                                    audio_chunk_samples=audio_data)
            msg = protocol.ExternalAudioStreamRequest(audio_stream_chunk=msg)
            yield msg
            await asyncio.sleep(0)

            # check if streaming is way ahead of audio playback time
            elapsed = time.time() - start_time
            expected_data_count = elapsed * params.framerate
            time_ahead = (curr_chunk * DEFAULT_FRAME_SIZE - expected_data_count) / params.framerate
            if time_ahead > 1.0:
                self.logger.debug("waiting %f to catchup chunk %f", time_ahead - 0.5, curr_chunk)
                await asyncio.sleep(time_ahead - 0.5)
            frames = frames - read_count
            curr_chunk += 1
            if curr_chunk == total_chunks:
                # last chunk:  time to stop stream
                msg = protocol.ExternalAudioStreamComplete()
                msg = protocol.ExternalAudioStreamRequest(audio_stream_complete=msg)

                yield msg
                await asyncio.sleep(0)

        reader.close()

        # Need the done message from the robot
        await self._done_event.wait()
        self._done_event.clear()

    async def _load_audio(self, source):
        """Loads an audio file from either a local file path or a URL into memory.

            This method supports both local file paths and remote URLs. If the source is a URL,
            it fetches the audio file via an HTTP request and loads it into memory. If the source
            is a local file path, it loads the audio directly from the file.

            :param source: The file path or URL to the audio file.
            :return: A pydub `AudioSegment` object containing the loaded audio data.
            :raises Exception: If the URL fetch fails (e.g., non-200 status code).

            Example usage for a local file:

            .. testcode file::

                audio = await robot.audio._load_audio('../examples/sounds/vector_alert.wav')

            Example usage for a URL:

            .. testcode url::

                audio = await robot.audio._load_audio('https://example.com/sounds/vector_alert.wav')
        """
        parsed_url = urlparse(source)
        is_url = bool(parsed_url.scheme in ('http', 'https'))

        try:
            if is_url:
                # Fetch the audio file from the URL
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(source, timeout=10) as response:
                            if response.status != 200:
                                raise VectorExternalAudioPlaybackException(
                                    f"Failed to download file from {source}. HTTP status: {response.status}"
                                )
                            audio_data = await response.read()

                        # Try loading the audio data into a pydub AudioSegment
                        try:
                            audio = AudioSegment.from_file(io.BytesIO(audio_data))
                        except Exception as e:
                            raise VectorExternalAudioPlaybackException(
                                f"Failed to process audio data from {source}. Error: {str(e)}"
                            )

                    except aiohttp.ClientError as e:
                        raise VectorExternalAudioPlaybackException(
                            f"Network error occurred while downloading file from {source}. Error: {str(e)}"
                        )
                    except asyncio.TimeoutError:
                        raise VectorExternalAudioPlaybackException(
                            f"Timed out while trying to download file from {source}."
                        )

            else:
                # Load the audio file directly from the local path
                try:
                    audio = AudioSegment.from_file(source)
                except FileNotFoundError:
                    raise VectorExternalAudioPlaybackException(f"Local file not found: {source}")
                except Exception as e:
                    raise VectorExternalAudioPlaybackException(
                        f"Failed to load local audio file from {source}. Error: {str(e)}"
                    )

        except VectorExternalAudioPlaybackException as e:
            self.logger.error(str(e))
            raise  # Re-raise the exception to be handled by the calling function

        return audio

    def _transcode_audio(self, audio):
        """Transcodes an audio file to WAV format (16000 Hz, 16 bits, 1 channel) and returns a BytesIO buffer.

            This method takes a `pydub.AudioSegment` object, transcodes it to the required format,
            and exports the result into a `BytesIO` buffer. The buffer is used to keep the audio
            data in memory for further processing or playback.

            :param audio: A `pydub.AudioSegment` object containing the audio data to transcode.
            :return: A `BytesIO` buffer containing the transcoded audio in WAV format.

            Example usage:

            .. testcode::

                transcoded_buffer = robot.audio._transcode_audio(audio_segment)
        """
        # Transcode the audio to the desired format
        transcoded_audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

        # Create a BytesIO buffer to hold the transcoded audio in WAV format
        audio_buffer = io.BytesIO()

        # Export the transcoded audio to the buffer in WAV format
        transcoded_audio.export(audio_buffer, format="wav")

        # Reset buffer position to the beginning
        audio_buffer.seek(0)

        return audio_buffer

    @on_connection_thread(requires_control=True)
    async def stream_audio(self, source, volume=50):
        """Plays audio using Vector's speakers.

            This method accepts either a local file path or a URL to an audio file.
            It automatically transcodes the audio to WAV format (16000 Hz, 16 bits, 1 channel)
            and streams it to Vector's speakers.

            Example usage for a local file:

            .. testcode file::

                import anki_vector

                with anki_vector.Robot() as robot:
                    robot.audio.stream_audio('../examples/sounds/vector_alert.wav')

            Example usage for a URL:

            .. testcode url::

                import anki_vector

                with anki_vector.Robot() as robot:
                    robot.audio.stream_audio('https://example.com/sounds/vector_alert.wav')

            :param source: The file path or URL to the audio file.
            :param volume: The audio playback level (0-100), where 0 is mute and 100 is maximum volume.
            :raises VectorExternalAudioPlaybackException: If the volume is out of range or another sound is already playing.
        """

        audio_buffer = None
        audio = None

        try:

            audio = await self._load_audio(source)
            audio_buffer = self._transcode_audio(audio)
            if audio:
                del audio

            # Play the transcoded file
            await self._play_audio(audio_buffer, volume)

        except Exception as e:
            raise VectorExternalAudioPlaybackException(f"Failed to stream audio from {source}. Reason: {str(e)}")

        finally:
            if audio_buffer:
                del audio_buffer

    async def _play_audio(self, audio_buffer, volume):

        """Plays the audio segment stored in memory on Vector's speaker.

            This method streams audio to Vector's speakers from a `BytesIO` buffer, which contains
            WAV audio data. It handles streaming in chunks, ensuring that playback occurs at the correct speed,
            and manages the state of audio playback. The method also checks that no other sound is playing before
            starting playback.

            :param audio_buffer: A `BytesIO` buffer containing the transcoded WAV audio data.
            :param volume: The audio playback level (0-100), where 0 is mute and 100 is maximum volume.
            :raises VectorExternalAudioPlaybackException: If another sound is already playing or if the volume is out of range.

            Example usage:

            .. testcode::

                await robot.audio._play_audio(transcoded_buffer, volume=75)
        """
        if self._is_active_event is None:
            self._is_active_event = asyncio.Event()

        if self._is_active_event.is_set():
            raise VectorExternalAudioPlaybackException("Cannot start audio when another sound is playing")

        if volume < 0 or volume > 100:
            raise VectorExternalAudioPlaybackException("Volume must be between 0 and 100")
        _file_reader, _file_params = self._open_file(audio_buffer)
        playback_error = None
        self._is_active_event.set()

        if self._done_event is None:
            self._done_event = asyncio.Event()

        try:
            async for response in self.grpc_interface.ExternalAudioStreamPlayback(
                    self._request_handler(_file_reader, _file_params, volume)):
                self.logger.info("ExternalAudioStream %s", MessageToString(response, as_one_line=True))
                response_type = response.WhichOneof("audio_response_type")
                if response_type == 'audio_stream_playback_complete':
                    playback_error = None
                elif response_type == 'audio_stream_buffer_overrun':
                    playback_error = response_type
                elif response_type == 'audio_stream_playback_failyer':
                    playback_error = response_type
                self._done_event.set()
        except asyncio.CancelledError:
            self.logger.debug('Audio Stream future was cancelled.')
        except futures.CancelledError:
            self.logger.debug('Audio Stream handler task was cancelled.')
        finally:
            self._is_active_event = None
            self._done_event = None

        if playback_error is not None:
            raise VectorExternalAudioPlaybackException(f"Error reported during audio playback {playback_error}")

    async def stream_robot_audio(self):
        """Public method to start the audio stream, yielding chunks for external processing."""
        if self._is_streaming:
            raise RuntimeError("Audio stream already running.")
        self._is_streaming = True

        try:
            async for audio_chunk in self._fetch_audio_feed():
                yield audio_chunk
        finally:
            self._is_streaming = False

    async def _fetch_audio_feed(self):
        """Fetches the audio feed from the robot."""
        request = self.grpc_interface.AudioFeedRequest()  # Send the request to initiate the audio feed
        async for response in self.conn.grpc_interface.AudioFeed(request):
            yield response.audio_data  # Extract audio data from the response

    def stop_audio_stream(self):
        """Stops the audio feed."""
        if not self._is_streaming:
            raise RuntimeError("Audio stream is not running.")
        self._is_streaming = False

    def is_streaming(self) -> bool:
        """Returns whether the audio stream is active."""
        return self._is_streaming