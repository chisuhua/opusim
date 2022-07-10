#!/bin/bash
# -*- coding: utf-8 -*-

set -e

build_dbg() {
	if [ ! -d "build" ]; then
		mkdir build && cd build && meson ..  && ninja -j8 || exit 1
	else
		cd build && ninja -j8 || exit 1
	fi
}

build_release() {
	if [ ! -d "build" ]; then
		mkdir build && cd build && meson .. --buildtype release || exit 1
	fi
	cd build && ninja -j8 || exit 1
}

if [ ! -z "$1" ]; then
	if [ $1 == lint ]; then
		echo start link check ...
	elif [ $1 == dbg ]; then
		build_dbg;
	elif [ $1 == rel ]; then
		build_release;
	fi
else
	build_dbg;
fi

