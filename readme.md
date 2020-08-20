# pytorch_mingw

  pytorch cpu_only on windows mingw (msys2/win10)


  Based on pytorch :

        https://github.com/pytorch/pytorch 


  msys2 : 
  
        http://www.msys2.org


  pytorch_mingw  install : 

      win10  cmd  console :
            
        cd pytorch_mingw
        
        md build

        cd build 

        cmake -G "MinGW Makefiles"  .. -DCMAKE_INSTALL_PREFIX="D:\mingw"

        mingw32-make  -j 8 
        
        mingw32-make  install

  
  examples  run :
      
     win10  cmd  console :

       set  PYTHONPATH=D:\mingw\python

       set  PATH=%PATH%;D:\mingw\lib 

       cd  pytorch_mingw\examples

       python  var_back.py


                                                       zhujiangmail@hotmail.com
                                                                   
                                                                2020.08.20    










