if (NOT __SHM_INCLUDED)
  set(__SHM_INCLUDED TRUE)

    # build directory
    set(SHM_PREFIX ${CMAKE_BINARY_DIR}/external/shm-prefix)  
    # install directory
    set(SHM_INSTALL ${CMAKE_BINARY_DIR}/external/tmp_install)

    set(SHM_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${SHM_EXTRA_COMPILER_FLAGS})
    set(SHM_C_FLAGS ${CMAKE_C_FLAGS} ${SHM_EXTRA_COMPILER_FLAGS})

    # depend on gflags if we're also building it
    if (SHM_EXTERNAL)
      set(SHM_DEPENDS shm)
    endif()

    ExternalProject_Add(shm_win
      PREFIX ${SHM_PREFIX}
      SOURCE_DIR  "${PROJECT_SOURCE_DIR}/torch/libshm_windows"
      INSTALL_DIR ${SHM_INSTALL}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                 -DCMAKE_INSTALL_PREFIX=${SHM_INSTALL}
                 -DCMAKE_C_FLAGS=${SHM_C_FLAGS}
                 -DCMAKE_CXX_FLAGS=${SHM_CXX_FLAGS}
      )

    set(SHM_FOUND TRUE)
    set(SHM_INCLUDE_DIRS ${SHM_INSTALL}/include)
    set(SHM_LIBRARIES ${SHMS_LIBRARIES} ${SHM_INSTALL}/lib/libshm.dll.a)
    set(SHM_LIBRARY_DIRS ${SHM_INSTALL}/lib)
    set(SHM_EXTERNAL TRUE)

    list(APPEND external_project_dependencies shm)
    
    INSTALL(FILES  ${CMAKE_BINARY_DIR}/external/tmp_install/lib/libshm.dll.a  DESTINATION lib)
    INSTALL(FILES  ${CMAKE_BINARY_DIR}/external/tmp_install/bin/libshm.dll  DESTINATION lib)

endif()

