#!/usr/bin/env -S uv run
# /// script
# dependencies = ["python-dotenv>=1.0.0"]
# ///
"""
SD Card watcher for automatic video download from wildlife cameras.
Monitors for USB SD card insertion, mounts them, and copies video files.
"""

import hashlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not available, use environment variables as-is
    pass

# Setup logging - we'll configure this after daemonization
logger = logging.getLogger(__name__)


class SDCardWatcher:
    def __init__(self):
        self.video_dir = Path(os.getenv("VIDEO_DIR", "./videos"))
        self.mount_base = Path(os.getenv("MOUNT_BASE", "/tmp/wildcams_mounts"))
        self.video_extensions = set(
            ext.strip().lower() for ext in os.getenv("VIDEO_EXTENSIONS", "mp4,mov,avi,mkv,m4v").split(",")
        )

        # Create directories
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.mount_base.mkdir(parents=True, exist_ok=True)

        # Track mounted devices
        self.mounted_devices: set[str] = set()
        self.running = True

        logger.info("🎥 SD Card watcher started")
        logger.info(f"📁 Video directory: {self.video_dir}")
        logger.info(f"🔍 Watching for extensions: {', '.join(self.video_extensions)}")

    def get_removable_devices(self) -> list[str]:
        """Get list of removable block devices (mounted and unmounted)."""
        devices = []
        try:
            result = subprocess.run(["lsblk", "-rno", "NAME,TYPE,RM,SIZE"], capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    name, dev_type, removable = parts[0], parts[1], parts[2]
                    # Look for removable partitions (not just disks)
                    if removable == "1" and dev_type == "part":
                        devices.append(f"/dev/{name}")
            return devices
        except subprocess.CalledProcessError:
            return []

    def get_mounted_removable_devices(self) -> list[tuple]:
        """Get list of mounted removable devices with their mount points."""
        mounted_devices = []
        try:
            # Get mounted filesystems
            result = subprocess.run(
                ["findmnt", "-rno", "SOURCE,TARGET,FSTYPE"], capture_output=True, text=True, check=True
            )

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    source, target = parts[0], parts[1]

                    # Skip if not a block device
                    if not source.startswith("/dev/"):
                        continue

                    # Check if it's a removable device
                    device_name = source.split("/")[-1]
                    # Strip partition number to get base device
                    base_device = "".join(c for c in device_name if not c.isdigit()) or device_name

                    try:
                        with open(f"/sys/block/{base_device}/removable") as f:
                            if f.read().strip() == "1":
                                mounted_devices.append((source, target))
                    except (FileNotFoundError, PermissionError):
                        # If we can't read removable status, check if it's in common mount locations
                        if any(target.startswith(path) for path in ["/media/", "/mnt/", "/run/media/"]):
                            mounted_devices.append((source, target))

            return mounted_devices
        except subprocess.CalledProcessError:
            return []

    def mount_device_userspace(self, device: str, max_retries: int = 5, retry_delay: float = 0.5) -> str | None:
        """Mount device using udisks2 (no sudo required).

        Retries if udisks hasn't registered the device yet, with exponential backoff.
        """
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ["udisksctl", "mount", "-b", device], capture_output=True, text=True, check=True
                )
                # Parse mount point from output like "Mounted /dev/sdc1 at /media/user/LABEL"
                for line in result.stdout.split("\n"):
                    if "Mounted" in line and "at" in line:
                        mount_point = line.split(" at ")[-1].strip()
                        logger.info(f"✅ Mounted {device} at {mount_point}")
                        return mount_point
                return None
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.strip() if e.stderr else ""

                # Transient errors that warrant retry
                if "Error looking up object" in stderr:
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"🔄 udisks not ready for {device}, retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                        continue
                    else:
                        logger.warning(f"❌ udisks never registered {device} after {max_retries} attempts")
                        return None

                # Permanent errors we can ignore
                if any(
                    msg in stderr
                    for msg in [
                        "is not a mountable filesystem",
                        "Already mounted",
                        "already mounted",
                        "Device is already mounted",
                    ]
                ):
                    logger.debug(f"⏭️  Skipping {device}: {stderr}")
                    return None

                # Other errors
                logger.warning(f"❌ Failed to mount {device}: {stderr}")
                return None

        return None

    def unmount_device_userspace(self, device: str) -> bool:
        """Unmount device using udisks2 (no sudo required)."""
        try:
            subprocess.run(["udisksctl", "unmount", "-b", device], check=True, capture_output=True)
            logger.info(f"🔄 Unmounted {device}")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"❌ Failed to unmount {device}: {e}")
            return False

    def is_likely_camera_card(self, mount_point: str) -> bool:
        """Check if a mounted device looks like a camera SD card."""
        mount_path = Path(mount_point)

        # Common camera directory structures
        camera_indicators = [
            "DCIM",  # Digital Camera Images
            "PRIVATE",  # Sony/other cameras
            "MP_ROOT",  # Some action cameras
            "AVCHD",  # AVCHD video format
        ]

        try:
            contents = [item.name.upper() for item in mount_path.iterdir() if item.is_dir()]
            return any(indicator in contents for indicator in camera_indicators)
        except (PermissionError, OSError):
            return False

    def find_video_files(self, path: Path) -> list[Path]:
        """Find all video files in the given path."""
        video_files = []
        for item in path.rglob("*"):
            if item.is_file() and item.suffix.lower().lstrip(".") in self.video_extensions:
                video_files.append(item)
        return sorted(video_files)

    def calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate MD5 hash of a file (fast and sufficient for duplicate detection)."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""

    def check_for_duplicate_hash(self, file_hash: str, original_name: str) -> tuple[bool, Path | None]:
        """
        Check if a hash already exists by calculating hashes of existing video files.
        Returns (is_duplicate, existing_file_path)
        """
        # Search through all existing video files and calculate their hashes
        for video_file in self.video_dir.rglob("*"):
            if video_file.is_file() and video_file.suffix.lower().lstrip(".") in self.video_extensions:
                try:
                    existing_hash = self.calculate_file_hash(video_file)
                    if existing_hash == file_hash:
                        logger.info(f"Found duplicate: {video_file.name} matches hash")
                        return True, video_file
                except Exception as e:
                    logger.warning(f"Could not calculate hash for {video_file}: {e}")

        return False, None

    def get_unique_filename(self, dest_dir: Path, original_name: str, file_hash: str) -> tuple[Path, bool]:
        """
        Get a unique filename, checking for hash collisions across ALL downloads.
        Returns (final_path, is_duplicate)
        """
        # First check if this hash already exists anywhere
        is_duplicate, existing_file = self.check_for_duplicate_hash(file_hash, original_name)
        if is_duplicate:
            return existing_file, True

        base_name = Path(original_name).stem
        extension = Path(original_name).suffix

        # Try the original filename first
        candidate_path = dest_dir / original_name

        # If file doesn't exist, use original name
        if not candidate_path.exists():
            return candidate_path, False

        # File exists but different hash (or no hash file), find new name
        counter = 1
        while True:
            new_name = f"{base_name}_{counter}{extension}"
            candidate_path = dest_dir / new_name

            if not candidate_path.exists():
                return candidate_path, False

            counter += 1
            if counter > 1000:  # Safety valve
                logger.error(f"Too many filename variations for {original_name}")
                return dest_dir / f"{base_name}_ERROR_{counter}{extension}", False

    def copy_videos(self, source_dir: Path, device_identifier: str) -> int:
        """Copy video files from source to destination with hash-based duplicate detection."""
        video_files = self.find_video_files(source_dir)

        if not video_files:
            logger.info("📂 No video files found on this device")
            return 0

        # Use the main video directory directly, no subdirectories
        dest_dir = self.video_dir

        logger.info(f"📥 Found {len(video_files)} video files, checking for duplicates...")

        copied_count = 0
        skipped_count = 0

        for i, video_file in enumerate(video_files, 1):
            try:
                # Calculate hash of source file
                file_hash = self.calculate_file_hash(video_file)
                if not file_hash:
                    logger.error(f"❌ Could not calculate hash for {video_file}, skipping")
                    continue

                # Check against existing files in ALL previous downloads
                relative_path = video_file.relative_to(source_dir)
                final_dest_file, is_duplicate = self.get_unique_filename(dest_dir, relative_path.name, file_hash)

                if is_duplicate:
                    logger.info(f"📂 Skipped duplicate: {video_file.name}")
                    skipped_count += 1
                    continue

                # Copy to the determined destination file
                final_dest_file.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"📋 Copied: {video_file.name}")
                shutil.copy2(video_file, final_dest_file)

                copied_count += 1

            except Exception as e:
                logger.error(f"❌ Failed to process {video_file}: {e}")

        # Summary
        if copied_count > 0 or skipped_count > 0:
            logger.info(f"✅ Copied: {copied_count}, Skipped: {skipped_count}")
            if copied_count > 0:
                logger.info(f"🎉 DOWNLOAD COMPLETE! Videos saved to: {dest_dir}")

        return copied_count

    def process_new_mount(self, device: str, mount_point: str):
        """Process a newly mounted device."""
        logger.info(f"📱 New mount detected: {device} at {mount_point}")

        # Check if it looks like a camera card
        if not self.is_likely_camera_card(mount_point):
            logger.info(f"⏭️  Skipping {mount_point} - doesn't look like a camera card")
            return

        logger.info(f"📷 Camera card detected at {mount_point}")

        # Create a device identifier from the mount point name
        device_identifier = Path(mount_point).name
        if not device_identifier or device_identifier in ["/", "."]:
            device_identifier = Path(device).name

        try:
            self.copy_videos(Path(mount_point), device_identifier)
        except Exception as e:
            logger.error(f"❌ Error processing {mount_point}: {e}")

    def process_unmounted_device(self, device: str):
        """Process an unmounted removable device by mounting it first."""
        logger.debug(f"🔍 Found unmounted removable device: {device}")

        # Try to mount it (will retry if udisks isn't ready yet)
        mount_point = self.mount_device_userspace(device)
        if not mount_point:
            # Couldn't mount - might be dirty, already mounted elsewhere, or not a filesystem
            return

        try:
            # Check if it looks like a camera card
            if not self.is_likely_camera_card(mount_point):
                logger.info(f"⏭️  Skipping {device} - doesn't look like a camera card")
                return

            logger.info(f"📷 Camera card detected at {mount_point}")

            # Create a device identifier
            device_identifier = Path(mount_point).name
            if not device_identifier or device_identifier in ["/", "."]:
                device_identifier = Path(device).name

            # Copy videos
            self.copy_videos(Path(mount_point), device_identifier)

        except Exception as e:
            logger.error(f"❌ Error processing {device}: {e}")
        finally:
            # Optionally unmount after processing (comment out if you want to leave mounted)
            # self.unmount_device_userspace(device)
            pass

    def _process_new_mount_wrapper(self, device: str, mount_point: str, processing_devices: set):
        """Wrapper for process_new_mount that handles cleanup."""
        try:
            self.process_new_mount(device, mount_point)
        finally:
            processing_devices.discard(device)

    def _process_unmounted_device_wrapper(self, device: str, processing_devices: set):
        """Wrapper for process_unmounted_device that handles cleanup."""
        try:
            self.process_unmounted_device(device)
        finally:
            processing_devices.discard(device)

    def watch_for_devices(self):
        """Main monitoring loop."""
        logger.info("👀 Watching for removable devices (mounted and unmounted)...")

        # Track what we saw last iteration to detect changes
        last_mounts = set()
        last_unmounted = set()

        # Track devices currently being processed to avoid double-processing
        processing_devices = set()

        while self.running:
            try:
                # Check mounted devices
                current_mounts = self.get_mounted_removable_devices()
                current_mount_set = set(current_mounts)

                # Process newly mounted devices (skip if already processing)
                new_mounts = current_mount_set - last_mounts
                for device, mount_point in new_mounts:
                    if device not in processing_devices:
                        processing_devices.add(device)
                        threading.Thread(
                            target=self._process_new_mount_wrapper,
                            args=(device, mount_point, processing_devices),
                            daemon=True,
                        ).start()

                # Check unmounted removable devices, but skip ones that are already mounted or processing
                all_removable = set(self.get_removable_devices())
                mounted_devices = {device for device, _ in current_mounts}
                unmounted_devices = all_removable - mounted_devices - processing_devices

                # Process newly appeared unmounted devices
                new_unmounted = unmounted_devices - last_unmounted
                for device in new_unmounted:
                    if device not in processing_devices:
                        processing_devices.add(device)
                        threading.Thread(
                            target=self._process_unmounted_device_wrapper,
                            args=(device, processing_devices),
                            daemon=True,
                        ).start()

                # Update for next iteration
                last_mounts = current_mount_set
                last_unmounted = unmounted_devices

                time.sleep(3)  # Check every 3 seconds

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)

        logger.info("🛑 SD Card watcher stopped")

    def stop(self):
        """Stop the watcher."""
        self.running = False


def setup_logging(daemon_mode=False, log_file=None):
    """Setup logging for both stdout and file output."""
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Clear any existing handlers
    logger.handlers.clear()
    logger.setLevel(log_level)

    # Always log to file if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Also log to stdout/stderr unless we're a full daemon
    if not daemon_mode:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        # Even in daemon mode, send logs to stdout so they appear in terminal
        # This works because our direnv setup captures the output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def daemonize():
    """Daemonize the process by forking twice."""
    try:
        # First fork
        pid = os.fork()
        if pid > 0:
            # Exit first parent
            sys.exit(0)
    except OSError as e:
        print(f"Fork #1 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    try:
        # Second fork
        pid = os.fork()
        if pid > 0:
            # Exit second parent
            sys.exit(0)
    except OSError as e:
        print(f"Fork #2 failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Don't redirect stdout/stderr - we want to see the logs!
    sys.stdout.flush()
    sys.stderr.flush()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("🛑 Received shutdown signal")
    watcher.stop()
    sys.exit(0)


def write_pid_file(pid_file: str):
    """Write the current process PID to a file."""
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SD Card Watcher Daemon")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--pid-file", default=".sd_watcher.pid", help="PID file location")
    parser.add_argument("--log-file", default=".sd_watcher.log", help="Log file location")
    args = parser.parse_args()

    # Setup logging first
    setup_logging(daemon_mode=args.daemon, log_file=args.log_file)

    # Daemonize if requested
    if args.daemon:
        daemonize()
        # Re-setup logging after daemonization to ensure stdout works
        setup_logging(daemon_mode=True, log_file=args.log_file)

    # Write PID file
    write_pid_file(args.pid_file)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start watcher
    watcher = SDCardWatcher()

    try:
        watcher.watch_for_devices()
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    finally:
        watcher.stop()
        # Clean up PID file
        try:
            os.unlink(args.pid_file)
        except OSError:
            pass
