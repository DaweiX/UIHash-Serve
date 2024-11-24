import argparse
import os.path
import subprocess
import time
from subprocess import check_output
from time import perf_counter, sleep
import platform
import re

from .util import init_logger, remove_sysnode, read_all_nodes
from .device import Device


class UICrawler:
    """Given a device, traverse apks on it and extract UIs at runtime.
    The parameters of this class are read from the command instead of
    a config file, because in practice we can launch multiple crawler
    in parallel (with different devices and apk ranges) to accelerate
    the dynamic data collecting.

    Attributes:
        device (Device): The device object we interact (via adb) with
    """

    def __init__(self, ip: str, log_level: str, log_path: str, 
                 input_apk: str, output_path: str, safe_package: str):
        self.handled_activity = set()
        # after 10 successive failure, we turns to the next app
        self._FAIL_TRY_TIMES = 10
        self._ip: str = ip
        self._logger = init_logger(log_level, log_path, "UICollect")
        self._device_type = "NOT_DEFINED"
        self.dom = None
        self._output_path = output_path
        self._apk = input_apk
        save_packages = safe_package.split(',')

        try:
            # the adb executable
            adb = "adb"
            # initialize a Device object and connect to the device
            self.device = Device(self._logger, save_packages=save_packages)
            retcode = self.device.connect(self._ip)
            if retcode == 0:
                devices: str = str(check_output([adb, "devices"]))
                if self._ip not in devices:
                    exit(1)
                    
            # ban auto rotation
            # self.device.run_shell("content insert --uri content://settings/tem "
            #                       "--bind name:s:accelerometer_rotation --bind "
            #                       "value:i:0")

            # first, do some clean up jobs
            self.device.stop_3rd_packages()
            self.device.uninstall_3rdpackages()

        except ConnectionError as ce:
            self._logger.error(f"ConnectionError: {ce}")
            exit(1)

    def dump_apk(self):
        """Install an apk, dump its UIs and then uninstall the app"""
        # install the app
        pkgs_old = self.device.get_3rdpackage_installed()
        self._logger.debug(f"Now install: {self._apk}")
        self.handled_activity.clear()
        t_pre_install = perf_counter()
        code = self.device.install_app(self._apk)
        t_post_install = perf_counter()
        if code == 0:
            pkgs_new = self.device.get_3rdpackage_installed()
            new_packages = list(set(pkgs_new).difference(pkgs_old))
            if not len(new_packages):
                self._logger.warning(f"Install fails: {self._apk}")
                return
            else:
                package_name = new_packages[0]
            self._logger.debug(f'Intall {package_name} finish '
                               f'in {t_post_install - t_pre_install}s')
            # dump ui
            self.dump_uis(package_name)
            # uninstall apk
            code = self.device.uninstall_package(package_name)
            if code != 0:
                self._logger.warning(f'Uninstall fail: {package_name}')
        else:
            self._logger.warning(f'Install fail: {self._apk}')

    def dump_uis(self, package_name: str):
        """Traverse UIs in an app

        Args:
            package_name (str): Package name declared in the manifest
        """
        self.device.stop_3rd_packages()
        self._logger.info(f"Current App: {package_name}")
        self.run_test(package_name)
        self._logger.info(f'UI collect finished: {package_name}')
        self.device.stop_3rd_packages()

    def run_test(self, package_name: str):
        """Explore UIs"""
        # start with main activity
        main_activity = self.device.start_activity(package_name, "", timeout=1)
        activity = self.device.get_current_activity()
        self._logger.info(f"Main activity: {main_activity}")
        self.handle_ui(main_activity)

        # we wait up to 5 seconds until an interactive view displays
        time_delay = 5
        ddl = time.time() + time_delay
        controls = []
        while len(controls) == 0 and time.time() < ddl:
            controls = []
            self.dom = self.device.get_current_dom()
            if self.dom is not None:
                read_all_nodes(controls, self.dom)
                controls = [self.get_dict(i) for i in controls]
                controls = [i for i in controls if len(i) > 0]
                self._logger.debug(f"Controls: {controls}")
                time.sleep(0.5)

        for c in controls:
            try:
                self._logger.debug(f"Starting: {main_activity}")
                # here, the activity name already contains the package name
                activity = self.device.start_activity(package_name, main_activity)
                self._logger.info(f"Current activity: {activity}")
                center = (c['c1'], c['c2'])
                self._logger.info(f"Now click: {c['name']}, center: {center}, text: {c['text']}")
                self.device.click(*center)
                time.sleep(0.8)
                self.handle_ui(activity)
                # back to the last activity
                self.device.press_key(4)
            except:
                continue
        
        alist = self.device.run_shell(f"dumpsys package {package_name} | "
                                      "grep -A 10 \"Activity\" | grep -E \"filter|exported\"", 
                                      get_output=True)
        if alist:
            activity_pattern = re.compile(r'([\w./]+) filter')
            activities = activity_pattern.findall(alist)
            for other_activity in activities:
                other_activity = other_activity.strip()
                if other_activity:   
                    self._logger.info(f"Exploring other activity: {other_activity}")
                    self.device.start_activity("", other_activity, timeout=1, sudo=True)
                    self.handle_ui(other_activity)


    @staticmethod
    def get_dict(node) -> dict:
        """Given a hierarchy node, extract the corresponding
        view's name, text, center if it is interactive

        Args:
            node: A node in hierarchy XML

        Returns:
            A dict. When the view is interactive, its 'c1' and 'c2'
              note the center coordinate, and 'name' and 'text' provide
              additional details. Otherwise, return an empty dict
        """
        _dict = dict()
        interact = node.getAttribute('clickable').startswith('t') \
                   or node.getAttribute('long-clickable').startswith('t') \
                   or node.getAttribute('checkable').startswith('t')
        if not interact:
            return _dict

        _dict['name'] = node.getAttribute('class')
        bounds = node.getAttribute('bounds').replace(']', '').split('[')
        if len(bounds) > 1:
            lt = bounds[1].split(',')
            rb = bounds[2].split(',')
            lt = [int(i) for i in lt]
            rb = [int(i) for i in rb]
            _dict['c1'] = int(lt[0] + 0.5 * (rb[0] - lt[0]))
            _dict['c2'] = int(lt[1] + 0.5 * (rb[1] - lt[1]))
        else:
            return dict()

        _dict['text'] = node.getAttribute('text')
        return _dict

    def handle_ui(self, activity: str):
        """Given an activity, save its UI info"""
        activity = activity.strip()
        self._logger.info(f"Handle activity: {activity}")
        if activity in self.handled_activity:
            self._logger.debug("Already seen this activity, skip.")
            return
        
        self.handled_activity.add(activity)

        self._logger.debug("Saving UI infomation")
        try:
            current_activity = self.device.get_current_activity()
            xml = self.device.dump_hierarchy()

            if xml is None:
                return

            if "package=\"android\"" in xml:
                self._logger.debug("Now in Android system UI, try to return to app")
                self.device.press_key(111)  # press esc
                sleep(0.5)
                self.device.press_key(111)  # press esc
                sleep(0.5)
                xml = self.device.dump_hierarchy()

            has_content, dom = remove_sysnode(xml)
            
            if not has_content:
                self._logger.debug('UI is without app content, continue')
                return
            
            self._logger.info("Taking UI screenshot")
            file_name = current_activity.replace('/', '-')
            if file_name.startswith("com.android.launcher"):
                self._logger.debug('Entering android desktop, return')
                return
            image_path = os.path.join(self._output_path, f"{file_name}.jpg")
            xml_path = os.path.join(self._output_path, f"{file_name}.xml")
            self.device.take_screenshot_screencap(image_path)
            self._logger.info(f'Screenshot saved in: {image_path}')

            with open(xml_path, 'w+', encoding='utf-8') as f:
                dom.writexml(f, addindent='  ', newl='\n')
                self._logger.info(f'Hierarchy saved in: {xml_path}')
            
            self.handled_activity.add(current_activity)

        except Exception as ex:
            self._logger.error(f'Exception when saving UI: {ex}')


def parse_arg_crawler():
    parser = argparse.ArgumentParser(description="Launch dynamic testing and collect UIs")
    parser.add_argument("input_apk", type=str, help="input apk file")
    parser.add_argument("output_path", type=str, help="apk feature folder")
    parser.add_argument("ip", type=str,
                        default="127.0.0.1:21523",
                        help="ip address of the android device")
    parser.add_argument("--log_level", "-ll",
                        help="logging level, default: info",
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARN", "ERROR"])
    parser.add_argument("--safe_package", "-sp", type=str, default="",
                        help="3rd packages need to keep on device (split by ,)")
    parser.add_argument("--log_path", "-lp", type=str, default="")
    _args = parser.parse_args()
    return _args


def main():
    # from subprocess import call
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/f", "/t", "/im", "adb"])
    else:
        subprocess.run(["killall", "adb"])
    args = parse_arg_crawler()
    u = UICrawler(ip=args.ip, log_level=args.log_level, log_path=args.log_path, 
                  input_apk=args.input_apk, output_path=args.output_path,
                  safe_package=args.safe_package)
    u.dump_apk()

if __name__ == '__main__':
    main()
