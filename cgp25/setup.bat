@echo off
set PATH=q:\mingw\MinGW\bin\;Q:\fitkit\QDevKit\bin;Q:\fitkit\mspgcc\bin;Q:\fitkit\Modelsim\win32;Q:\fitkit\XilinxISE\ISE\bin\nt;Q:\fitkit\XilinxISE\common\bin\nt;Q:\fitkit\XilinxISE\PlanAhead\bin;Q:\fitkit\Precision\Mgc_home\bin;Q:\fitkit\CatapultC.new\Mgc_home\bin;%PATH%

echo Type "gmake -C cgp" to compile the CGP
echo Type "gmake -C cgp run" to compile and run CGP
echo Type "gmake -C tab2h" to compile the tab2h utility

cmd