# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: anki_vector/messaging/settings.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from anki_vector.messaging import response_status_pb2 as anki__vector_dot_messaging_dot_response__status__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n$anki_vector/messaging/settings.proto\x12\x1e\x41nki.Vector.external_interface\x1a+anki_vector/messaging/response_status.proto\"\xe2\x02\n\x13RobotSettingsConfig\x12\x15\n\rclock_24_hour\x18\x01 \x01(\x08\x12;\n\teye_color\x18\x02 \x01(\x0e\x32(.Anki.Vector.external_interface.EyeColor\x12\x18\n\x10\x64\x65\x66\x61ult_location\x18\x03 \x01(\t\x12\x16\n\x0e\x64ist_is_metric\x18\x04 \x01(\x08\x12\x0e\n\x06locale\x18\x05 \x01(\t\x12=\n\rmaster_volume\x18\x06 \x01(\x0e\x32&.Anki.Vector.external_interface.Volume\x12\x1a\n\x12temp_is_fahrenheit\x18\x07 \x01(\x08\x12\x11\n\ttime_zone\x18\x08 \x01(\t\x12G\n\x0f\x62utton_wakeword\x18\t \x01(\x0e\x32..Anki.Vector.external_interface.ButtonWakeWord\"u\n\x15\x41\x63\x63ountSettingsConfig\x12\x19\n\x0f\x64\x61ta_collection\x18\x01 \x01(\x08H\x00\x12\x14\n\napp_locale\x18\x02 \x01(\tH\x01\x42\x17\n\x15oneof_data_collectionB\x12\n\x10oneof_app_locale\"N\n\x16UserEntitlementsConfig\x12\x1a\n\x10kickstarter_eyes\x18\x01 \x01(\x08H\x00\x42\x18\n\x16oneof_kickstarter_eyes\"[\n\x04Jdoc\x12\x13\n\x0b\x64oc_version\x18\x01 \x01(\x04\x12\x13\n\x0b\x66mt_version\x18\x02 \x01(\x04\x12\x17\n\x0f\x63lient_metadata\x18\x03 \x01(\t\x12\x10\n\x08json_doc\x18\x04 \x01(\t\"{\n\tNamedJdoc\x12;\n\tjdoc_type\x18\x01 \x01(\x0e\x32(.Anki.Vector.external_interface.JdocType\x12\x31\n\x03\x64oc\x18\x02 \x01(\x0b\x32$.Anki.Vector.external_interface.Jdoc\"P\n\x10PullJdocsRequest\x12<\n\njdoc_types\x18\x01 \x03(\x0e\x32(.Anki.Vector.external_interface.JdocType\"\x93\x01\n\x11PullJdocsResponse\x12>\n\x06status\x18\x01 \x01(\x0b\x32..Anki.Vector.external_interface.ResponseStatus\x12>\n\x0bnamed_jdocs\x18\x02 \x03(\x0b\x32).Anki.Vector.external_interface.NamedJdoc\"^\n\x15UpdateSettingsRequest\x12\x45\n\x08settings\x18\x01 \x01(\x0b\x32\x33.Anki.Vector.external_interface.RobotSettingsConfig\"\xc5\x01\n\x16UpdateSettingsResponse\x12>\n\x06status\x18\x01 \x01(\x0b\x32..Anki.Vector.external_interface.ResponseStatus\x12\x38\n\x04\x63ode\x18\x02 \x01(\x0e\x32*.Anki.Vector.external_interface.ResultCode\x12\x31\n\x03\x64oc\x18\x03 \x01(\x0b\x32$.Anki.Vector.external_interface.Jdoc\"o\n\x1cUpdateAccountSettingsRequest\x12O\n\x10\x61\x63\x63ount_settings\x18\x01 \x01(\x0b\x32\x35.Anki.Vector.external_interface.AccountSettingsConfig\"\xcc\x01\n\x1dUpdateAccountSettingsResponse\x12>\n\x06status\x18\x01 \x01(\x0b\x32..Anki.Vector.external_interface.ResponseStatus\x12\x38\n\x04\x63ode\x18\x02 \x01(\x0e\x32*.Anki.Vector.external_interface.ResultCode\x12\x31\n\x03\x64oc\x18\x03 \x01(\x0b\x32$.Anki.Vector.external_interface.Jdoc\"r\n\x1dUpdateUserEntitlementsRequest\x12Q\n\x11user_entitlements\x18\x01 \x01(\x0b\x32\x36.Anki.Vector.external_interface.UserEntitlementsConfig\"\xcd\x01\n\x1eUpdateUserEntitlementsResponse\x12>\n\x06status\x18\x01 \x01(\x0b\x32..Anki.Vector.external_interface.ResponseStatus\x12\x38\n\x04\x63ode\x18\x02 \x01(\x0e\x32*.Anki.Vector.external_interface.ResultCode\x12\x31\n\x03\x64oc\x18\x03 \x01(\x0b\x32$.Anki.Vector.external_interface.Jdoc\"L\n\x0cJdocsChanged\x12<\n\njdoc_types\x18\x01 \x03(\x0e\x32(.Anki.Vector.external_interface.JdocType*%\n\nApiVersion\x12\x0b\n\x07INVALID\x10\x00\x12\n\n\x06LATEST\x10\x01*R\n\x06Volume\x12\x08\n\x04MUTE\x10\x00\x12\x07\n\x03LOW\x10\x01\x12\x0e\n\nMEDIUM_LOW\x10\x02\x12\n\n\x06MEDIUM\x10\x03\x12\x0f\n\x0bMEDIUM_HIGH\x10\x04\x12\x08\n\x04HIGH\x10\x05*e\n\x08JdocType\x12\x12\n\x0eROBOT_SETTINGS\x10\x00\x12\x18\n\x14ROBOT_LIFETIME_STATS\x10\x01\x12\x14\n\x10\x41\x43\x43OUNT_SETTINGS\x10\x02\x12\x15\n\x11USER_ENTITLEMENTS\x10\x03*;\n\x11JdocResolveMethod\x12\x11\n\rPUSH_TO_CLOUD\x10\x00\x12\x13\n\x0fPULL_FROM_CLOUD\x10\x01*\xb5\x01\n\x0cRobotSetting\x12\x11\n\rclock_24_hour\x10\x00\x12\r\n\teye_color\x10\x01\x12\x14\n\x10\x64\x65\x66\x61ult_location\x10\x02\x12\x12\n\x0e\x64ist_is_metric\x10\x03\x12\n\n\x06locale\x10\x04\x12\x11\n\rmaster_volume\x10\x05\x12\x16\n\x12temp_is_fahrenheit\x10\x06\x12\r\n\ttime_zone\x10\x07\x12\x13\n\x0f\x62utton_wakeword\x10\x08*\xab\x01\n\x08\x45yeColor\x12\x11\n\rTIP_OVER_TEAL\x10\x00\x12\x12\n\x0eOVERFIT_ORANGE\x10\x01\x12\x12\n\x0eUNCANNY_YELLOW\x10\x02\x12\x13\n\x0fNON_LINEAR_LIME\x10\x03\x12\x18\n\x14SINGULARITY_SAPPHIRE\x10\x04\x12\x19\n\x15\x46\x41LSE_POSITIVE_PURPLE\x10\x05\x12\x1a\n\x16\x43ONFUSION_MATRIX_GREEN\x10\x06*K\n\x0e\x42uttonWakeWord\x12\x1e\n\x1a\x42UTTON_WAKEWORD_HEY_VECTOR\x10\x00\x12\x19\n\x15\x42UTTON_WAKEWORD_ALEXA\x10\x01*5\n\x0e\x41\x63\x63ountSetting\x12\x13\n\x0f\x44\x41TA_COLLECTION\x10\x00\x12\x0e\n\nAPP_LOCALE\x10\x01*\'\n\x0fUserEntitlement\x12\x14\n\x10KICKSTARTER_EYES\x10\x00*A\n\nResultCode\x12\x15\n\x11SETTINGS_ACCEPTED\x10\x00\x12\x1c\n\x18\x45RROR_UPDATE_IN_PROGRESS\x10\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'anki_vector.messaging.settings_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _APIVERSION._serialized_start=2141
  _APIVERSION._serialized_end=2178
  _VOLUME._serialized_start=2180
  _VOLUME._serialized_end=2262
  _JDOCTYPE._serialized_start=2264
  _JDOCTYPE._serialized_end=2365
  _JDOCRESOLVEMETHOD._serialized_start=2367
  _JDOCRESOLVEMETHOD._serialized_end=2426
  _ROBOTSETTING._serialized_start=2429
  _ROBOTSETTING._serialized_end=2610
  _EYECOLOR._serialized_start=2613
  _EYECOLOR._serialized_end=2784
  _BUTTONWAKEWORD._serialized_start=2786
  _BUTTONWAKEWORD._serialized_end=2861
  _ACCOUNTSETTING._serialized_start=2863
  _ACCOUNTSETTING._serialized_end=2916
  _USERENTITLEMENT._serialized_start=2918
  _USERENTITLEMENT._serialized_end=2957
  _RESULTCODE._serialized_start=2959
  _RESULTCODE._serialized_end=3024
  _ROBOTSETTINGSCONFIG._serialized_start=118
  _ROBOTSETTINGSCONFIG._serialized_end=472
  _ACCOUNTSETTINGSCONFIG._serialized_start=474
  _ACCOUNTSETTINGSCONFIG._serialized_end=591
  _USERENTITLEMENTSCONFIG._serialized_start=593
  _USERENTITLEMENTSCONFIG._serialized_end=671
  _JDOC._serialized_start=673
  _JDOC._serialized_end=764
  _NAMEDJDOC._serialized_start=766
  _NAMEDJDOC._serialized_end=889
  _PULLJDOCSREQUEST._serialized_start=891
  _PULLJDOCSREQUEST._serialized_end=971
  _PULLJDOCSRESPONSE._serialized_start=974
  _PULLJDOCSRESPONSE._serialized_end=1121
  _UPDATESETTINGSREQUEST._serialized_start=1123
  _UPDATESETTINGSREQUEST._serialized_end=1217
  _UPDATESETTINGSRESPONSE._serialized_start=1220
  _UPDATESETTINGSRESPONSE._serialized_end=1417
  _UPDATEACCOUNTSETTINGSREQUEST._serialized_start=1419
  _UPDATEACCOUNTSETTINGSREQUEST._serialized_end=1530
  _UPDATEACCOUNTSETTINGSRESPONSE._serialized_start=1533
  _UPDATEACCOUNTSETTINGSRESPONSE._serialized_end=1737
  _UPDATEUSERENTITLEMENTSREQUEST._serialized_start=1739
  _UPDATEUSERENTITLEMENTSREQUEST._serialized_end=1853
  _UPDATEUSERENTITLEMENTSRESPONSE._serialized_start=1856
  _UPDATEUSERENTITLEMENTSRESPONSE._serialized_end=2061
  _JDOCSCHANGED._serialized_start=2063
  _JDOCSCHANGED._serialized_end=2139
# @@protoc_insertion_point(module_scope)
