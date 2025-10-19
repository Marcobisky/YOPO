# Windows Docker Setup Guide

## For Windows Users Only

### Prerequisites
1. **Docker Desktop** - Install from https://www.docker.com/products/docker-desktop/
2. **VcXsrv** (X Server for Windows) - Download from https://sourceforge.net/projects/vcxsrv/

### One-Time Setup

1. Install Docker Desktop and VcXsrv
2. Start VcXsrv with XLaunch:
   - Multiple windows
   - Display number: 0
   - Start no client
   - **IMPORTANT: Check "Disable access control"**
   - Save configuration for future use

### Running the Project

Simply double-click `RUN_WINDOWS.bat` and follow the instructions.

The script will:
1. Build the Docker container (first time only)
2. Run the tracking program
3. Display live visualization windows
4. Generate GIF files in the `gifs/` folder

### Troubleshooting

**Problem: No windows appear**
- Make sure VcXsrv is running
- Ensure "Disable access control" is checked in VcXsrv settings

**Problem: Docker error**
- Make sure Docker Desktop is running
- Check that virtualization is enabled in BIOS

**Problem: Build fails**
- Check your internet connection
- Try running the .bat file as Administrator
