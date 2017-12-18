echo off
cd ../../../out/production/StdLibExtra
javah KKH.StdLib.ailib
echo Calling javah on the java class...
timeout /t 3
copy KKH_StdLib_ailib.h "../../../JNI/ailib/ailib_JNI/KKH_StdLib_ailib.h"
echo The file "KKH_StdLib_ailib.h" generated in this folder.
pause