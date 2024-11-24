"""Android device object"""

import os.path
import subprocess
import logging
import time
from subprocess import CalledProcessError, check_call, check_output
from typing import Union, Tuple
from enum import Enum
import re

from .util import remove_sysnode


class Device:
    """An Android device. This class provides APIs to interact with,
    transmit message to, and fetch status from teh device"""

    class Attr(Enum):
        model = "ro.product.model"
        sdk_version = "ro.build.version.sdk"
        cpu_abi = "ro.product.cpu.abi"

    def __init__(self, logger: logging.Logger,
                 device_name: str = "",
                 save_packages: list = None):
        self.logger = logger
        self.connected = False
        self.device_serial = device_name
        self.display = {}
        self.save_packages = save_packages

    @staticmethod
    def get_device_name_by_index(device_index: int = 0) -> Union[str, None]:
        """Get serial by an index in ADB devices"""
        devices = str(check_output("adb devices"))
        i = -2
        for line in devices.split('\n'):
            i += 1
            if i == device_index:
                return line.split('\t')[0].strip()
        return None

    @staticmethod
    def ip_connected(device_ip: str) -> bool:
        """Detect whether an IP is connected with ADB"""
        devices = str(check_output("adb devices"))
        for line in devices.split('\n'):
            if device_ip in line:
                return True
        return False

    def connect(self, device_ip: str) -> int:
        """Connect to an IP via ADB and report the result

        Args:
            device_ip (str): Device IP

        Returns:
            Return code. 0 for success, otherwise the return code
              of the CalledProcessError

        Raises:
            CalledProcessError: Raise when the command fails

        """
        try:
            self.logger.debug(f"Connect to {device_ip} via adb...")
            check_call(["adb", "root"])
            check_call(["adb", "connect", device_ip])
            devices: str = str(check_output(["adb", "devices"]))
            if device_ip in devices:
                self.connected = True
                self.device_serial = device_ip
            self.logger.debug(f"ADB connected to {device_ip}")
            return 0
        except CalledProcessError as e:
            self.logger.error(f"Cmd {e.cmd} failed: {e.output}")
            return e.returncode

    def wait_device_ready(self) -> int:
        """Wait for the device until it is online

        Raises:
            CalledProcessError: Raise when the ADB command fails
              or something is wrong with the device
        """
        self.logger.debug("waiting device...")
        try:
            check_call(["adb", "-s", self.device_serial, "wait-for-device"])
            self.logger.debug("ready")
            return 0
        except CalledProcessError as e:
            self.logger.error(f"Cmd {e.cmd} failed: {e.output}")
            return e.returncode

    def run_shell(self, shell_cmd: str,
                  get_output: bool = False, sudo: bool = False) -> Union[int, str]:
        """Push ADB shell commands to the device

        Args:
            shell_cmd (str): A command line
            get_output (bool): If shell output is required, set it True

        Returns:
            An int indicating the return code, or the shell output (str)
        """
        cmd = ["adb", "-s", self.device_serial, "shell"]
        if sudo:
            cmd.extend(["su", "0"])
        cmd.extend(shell_cmd.split(' '))
        cmd = [c for c in cmd if len(c)]
        cmd_str = ' '.join(cmd)
        try:
            self.logger.debug(f"Run: {cmd_str}")
            if not get_output:
                subprocess.run(cmd, timeout=5, capture_output=False, shell=False)
                return 0
            else:
                # timeout may not work with some adb versions (e.g., the memuc adb)
                process = subprocess.run(cmd, timeout=5, capture_output=True, shell=False)

                result = process.stdout.decode()
                if len(result) == 0:
                    result = process.stderr.decode()
                return result
            
        except CalledProcessError as e:
            self.logger.error(f"cmd {e.cmd} failed: {e}")
            devices: str = str(check_output(["adb", "devices"]))
            if self.device_serial not in devices:
                self.connected = False
                raise ValueError(f"device {self.device_serial} is offline")
            else:
                return e.returncode
        except subprocess.TimeoutExpired:
            self.logger.info("Restart ADB server...")
            subprocess.run(["adb", "-s", self.device_serial, "kill-server"])
            self.logger.info("ADB Server killed")
            subprocess.run(["adb", "-s", self.device_serial, "start-server"])
            subprocess.run(["adb", "connect", self.device_serial])
            self.logger.info("ADB Server restarted")
            self.escape_stuck()
            self.run_shell(shell_cmd)

    def get_current(self) -> Tuple[str, str]:
        """Get current package and activity"""
        c = self.run_shell("dumpsys window windows | grep 'Current'",
                           get_output=True)
        c = c.split(' ')[-1].replace('}', '')
        if not c:
            return '', ''
        package, activity = c.split('/')
        return package, activity

    def press_key(self, keycode: Union[str, int]):
        self.run_shell(f"input keyevent {keycode}")

    def get_device_attribute(self, key: str) -> str:
        """Get certain attributes (like sdk version) of the device"""
        if key in dir(self):
            return self.__getattribute__(key)
        ro_key = Device.Attr[key].value
        value = self.run_shell(f"getprop {ro_key}", get_output=True)
        value = value.strip()
        self.__setattr__(key, value)
        return value

    def get_sdk_version(self):
        v = self.get_device_attribute(Device.Attr.sdk_version.name)
        if "\n" in v:
            v = v.split("\n")[-1]
        return v

    def get_model(self):
        return self.get_device_attribute(Device.Attr.model.name)

    def get_abi(self):
        abi = self.get_device_attribute(Device.Attr.cpu_abi.name)
        if "\n" in abi:
            abi = abi.split("\n")[-1]
        return abi

    def get_3rdpackage_installed(self) -> list:
        """list all the installed 3rd-party apps"""
        output = self.run_shell("pm list packages -3", get_output=True)
        packages = re.findall(r'package:([^\s]+)', output)
        return list(packages)

    def get_3rdpackage_running(self) -> list:
        """list all the running 3rd-party apps"""
        packages = self.get_3rdpackage_installed()
        process_names = re.findall(r'([^\s]+)$',
                                   self.run_shell("ps; ps -A",
                                                  get_output=True),
                                   re.M)
        return list(set(packages).intersection(process_names))

    def stop_package(self, name: str):
        """Stop a package"""
        code = self.run_shell(f"am force-stop {name}")
        if code != 0:
            self.logger.warning(f"stop {name} fails")

    def stop_3rd_packages(self):
        """Stop all 3rd-party packages"""
        targets = self.get_3rdpackage_running()
        for t in targets:
            self.stop_package(t)
        self.logger.info("All the 3rd packages are stopped")

    def clear_package_data(self, name: str):
        """Clear data of a package"""
        self.run_shell(f"pm clear {name}")

    def uninstall_package(self, package_name: str) -> int:
        """Uninstall a package

        Returns:
            A return code
        """
        code = self.run_shell(f"pm uninstall {package_name}")
        if code == 0:
            self.logger.info(f"Uninstalled: {package_name}")
        return code

    def uninstall_3rdpackages(self):
        """Uninstall all 3rd-party packages"""
        self.logger.info("Uninstalling 3rd packages")
        for a in self.get_3rdpackage_installed():
            if a in self.save_packages:
                self.logger.debug(f"Uninstall skip: {a}")
            else:
                self.logger.debug(f"Uninstall: {a}")
                self.uninstall_package(a)
        self.logger.debug("All the 3rd packages are uninstalled")

    def get_package(self):
        return self.run_shell("dumpsys window | grep mCurrentFocus | "
                              "awk -F ' ' '{print $3}' | cut -d '/' -f 1", 
                              get_output=True)

    def start_activity(self, package_name: str,
                       activity: str, timeout: int = 0.8,
                       sudo: bool = False) -> Union[str, None]:
        """Try to launch an app with certain activity

        Args:
            package_name (str): The package expected to launch. If it is empty,
              then the activity name will be used instead
            activity (str): The activity we try to access. If it is empty,
              then try to start the package from its default (main) activity.
              We use monkey to access the main activity
            timeout (float): A delay to wait for the activity

        Returns:
            None if fail, an activity name if success
        """
        name = activity
        if len(package_name) > 0:
            name = f"{package_name}/{activity}"
        if name.count("/") > 1:
            name = "/".join(name.split("/")[-2:])
        if len(activity) > 0:
            cmd = f"am start -n {name}"
            self.run_shell(cmd, sudo=sudo)
        else:
            # get main activity
            output = self.run_shell(f"dumpsys package {package_name}" 
                                    f"| grep -A 1 MAIN | grep {package_name}"
                                    "| awk '{print $2}'" , get_output=True)
            self.run_shell(f"am start -n {output}", sudo=sudo)
            return output
        ddl = time.time() + timeout
        while time.time() < ddl:
            current_package = self.get_package()
            if current_package == package_name:
                break
            time.sleep(.2)
        time.sleep(.2)
        current_package = self.get_package()
        if current_package is None:
            self.logger.warning(f"Cannot get focus window")
            return ''
        if len(current_package) == 0:
            self.logger.warning(f"Cannot get focus window")
            return ''
        if current_package != package_name:
            self.logger.warning(f"Cannot launch app from activity {activity}")
            return ''

        return self.get_current_activity()       

    def install_app(self, apk_path: str) -> int:
        """Install an app on the device

        Args:
            apk_path (str): An apk file

        Returns:
            A return code
        """
        process = subprocess.run(["adb", "-s", self.device_serial, "install",
                                  "-r", "-d", "-g", apk_path], capture_output=True)
        return process.returncode

    def get_display_info(self):
        """Get device display information. This function will
        also get and set width/height of the device object. Value for
        the key named rotation is set to 0 (0), 1 (90), 2 (180), or 3 (270).
        All the results will be saved in `device.display` as well.

        Returns:
            A dict with three keys: orientation, width, and height
        """
        display_re = re.compile(
            r'.*DisplayViewport{valid=true, '
            r'.*orientation=(?P<orientation>\d+), '
            r'.*deviceWidth=(?P<width>\d+), '
            r'deviceHeight=(?P<height>\d+).*'
        )
        output = self.run_shell("dumpsys display", get_output=True).splitlines()
        for line in output:
            m = display_re.search(line, 0)
            if not m:
                continue
            self.display['orientation'] = m.group('orientation')
            self.display['width'] = m.group('width')
            self.display['height'] = m.group('height')
        return self.display

    def pull(self, src: str, dst: str):
        """Pull a file from the device to localhost"""
        try:
            check_call(["adb", "-s", self.device_serial, "pull", src, dst])
            self.logger.debug(f"Pull file: {src} -> {dst}")
        except CalledProcessError:
            return

    def push(self, src: str, dst: str):
        """Push a file to the device from localhost"""
        try:
            check_call(["adb", "-s", self.device_serial, "push", src, dst])
        except CalledProcessError:
            return

    def take_screenshot_screencap(self, save_path: str) -> bool:
        """Take a screenshot via ADB"""
        remote_image_path = "/data/local/tmp/screenshot.png"
        self.run_shell(f"screencap -p {remote_image_path}")
        self.pull(remote_image_path, save_path)
        return True

    def click(self, x, y):
        """Click (tap) on the device screen

        Args:
            x: Horizontal coordinate
            y: Vertical coordinate
        """
        self.run_shell(f"input tap {x} {y}")

    def dump_hierarchy(self) -> str:
        """Dump and output the current hierarchy via uiautomator"""
        tmp_path = "tmp11.xml"
        xml_path = self.run_shell("uiautomator dump", get_output=True)
        if xml_path is None:
            return None
        xml_path = xml_path.split(' ')[-1].strip()
        self.logger.debug(f"Hieracrchy path: {xml_path}")
        if xml_path.endswith('.xml'):
            self.pull(xml_path, tmp_path)
            if not os.path.exists(tmp_path):
                return None
            
            with open(tmp_path, mode='r', encoding='utf-8') as xmlf:
                xml = xmlf.read()
                os.remove(tmp_path)

            self.logger.debug(f"XML length: {len(xml)}")
            return xml
        
    def get_current_dom(self):
        xml = self.dump_hierarchy()
        if xml is not None:
            _, dom = remove_sysnode(xml)
            return dom
        return None

    def get_current_activity(self):
        """Get current activity"""
        r = self.run_shell("dumpsys activity activities", get_output=True)
        activity_line_re = re.compile(r'\*\s*Hist\s*#\d+:\s*ActivityRecord\{[^ ]+\s*[^ ]+\s*([^ ]+)\s*t(\d+)}')
        m = activity_line_re.search(r)
        if m:
            return m.group(1)
        return "None"

    def escape_stuck(self):
        """In some cases, the Andoid OS may be freezed (e.g., by
        system built-in dialogs. Then, try to flee from the stuck situation"""
        for _ in range(12):  # greater than max failure
            self.press_key(111)
            time.sleep(0.3)
        time.sleep(1)