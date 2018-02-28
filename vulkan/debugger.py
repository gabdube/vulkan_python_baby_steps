# -*- coding: utf-8 -*-
"""
A simple vulkan debugger. It prints vulkan warning and error in the stdout
"""

from . import vk
from ctypes import byref
from enum import IntFlag
import weakref


class Debugger(object):
    """
     A high level wrapper over the debug report vulkan extension. When an error is catched, it is printed somewhere
    """

    def __init__(self, api, instance):
        self.api = weakref.ref(api)
        self.instance = instance
        self.debug_report_callback = None
        self.callback_fn = None
        self.callbacks = [print]

        f = DebugReportFlags
        self.report_flags = f.Information | f.Warning | f.PerformanceWarning | f.Error

    @property
    def running(self):
        return self.debug_report_callback is not None

    def format_debug(self, flags, object_type, object, location, message_code, layer, message, user_data):
        message_type = DebugReportFlags(flags).name
        message = message[::].decode()
        full_message = f"{message_type}: {message}"

        for callback in self.callbacks:
            callback(full_message)

        return 0

    def start(self):
        """ Start the debugger """
        if self.running:
            self.stop()

        api, instance = self.api(), self.instance
        callback_fn = vk.FnDebugReportCallbackEXT(lambda *args: self.format_debug(*args))
        create_info = vk.DebugReportCallbackCreateInfoEXT(
            type = vk.STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            next = None,
            flags = self.report_flags,
            callback = callback_fn,
            user_data = None
        )

        debug_report_callback = vk.DebugReportCallbackEXT(0)
        result = api.CreateDebugReportCallbackEXT(
            instance, byref(create_info), None, byref(debug_report_callback)
        )

        if result != vk.SUCCESS:
            raise RuntimeError(f"Failed to start the vulkan debug report: {result}")

        self.callback_fn = callback_fn
        self.debug_report_callback = debug_report_callback

    def stop(self):
        """ Stop the debugger """
        if not self.running:
            return

        api, instance = self.api(), self.instance
        api.DestroyDebugReportCallbackEXT(instance, self.debug_report_callback, None)
        self.debug_report_callback = None
        self.callback_fn = None


class DebugReportFlags(IntFlag):
    Information = vk.DEBUG_REPORT_INFORMATION_BIT_EXT
    Warning = vk.DEBUG_REPORT_WARNING_BIT_EXT
    PerformanceWarning = vk.DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT
    Error = vk.DEBUG_REPORT_ERROR_BIT_EXT
    Debug = vk.DEBUG_REPORT_DEBUG_BIT_EXT
