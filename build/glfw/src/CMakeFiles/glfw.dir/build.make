# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xiuyan/libigl-test-project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xiuyan/libigl-test-project/build

# Include any dependencies generated for this target.
include glfw/src/CMakeFiles/glfw.dir/depend.make

# Include the progress variables for this target.
include glfw/src/CMakeFiles/glfw.dir/progress.make

# Include the compile flags for this target's objects.
include glfw/src/CMakeFiles/glfw.dir/flags.make

# Object files for target glfw
glfw_OBJECTS =

# External object files for target glfw
glfw_EXTERNAL_OBJECTS = \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/context.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/init.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/input.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/monitor.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/vulkan.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/window.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/x11_init.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/x11_monitor.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/x11_window.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/xkb_unicode.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/linux_joystick.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/posix_time.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/posix_tls.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/glx_context.c.o" \
"/home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw_objects.dir/egl_context.c.o"

glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/context.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/init.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/input.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/monitor.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/vulkan.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/window.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/x11_init.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/x11_monitor.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/x11_window.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/xkb_unicode.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/linux_joystick.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/posix_time.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/posix_tls.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/glx_context.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw_objects.dir/egl_context.c.o
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw.dir/build.make
glfw/src/libglfw3.a: glfw/src/CMakeFiles/glfw.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xiuyan/libigl-test-project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking C static library libglfw3.a"
	cd /home/xiuyan/libigl-test-project/build/glfw/src && $(CMAKE_COMMAND) -P CMakeFiles/glfw.dir/cmake_clean_target.cmake
	cd /home/xiuyan/libigl-test-project/build/glfw/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/glfw.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
glfw/src/CMakeFiles/glfw.dir/build: glfw/src/libglfw3.a

.PHONY : glfw/src/CMakeFiles/glfw.dir/build

glfw/src/CMakeFiles/glfw.dir/requires:

.PHONY : glfw/src/CMakeFiles/glfw.dir/requires

glfw/src/CMakeFiles/glfw.dir/clean:
	cd /home/xiuyan/libigl-test-project/build/glfw/src && $(CMAKE_COMMAND) -P CMakeFiles/glfw.dir/cmake_clean.cmake
.PHONY : glfw/src/CMakeFiles/glfw.dir/clean

glfw/src/CMakeFiles/glfw.dir/depend:
	cd /home/xiuyan/libigl-test-project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xiuyan/libigl-test-project /home/xiuyan/libigl/external/nanogui/ext/glfw/src /home/xiuyan/libigl-test-project/build /home/xiuyan/libigl-test-project/build/glfw/src /home/xiuyan/libigl-test-project/build/glfw/src/CMakeFiles/glfw.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : glfw/src/CMakeFiles/glfw.dir/depend

