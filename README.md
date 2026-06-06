# StrikeX 2026 PC

Windows-side software for the StrikeX Autonomous Air Defense System.

## Current Status

- UDP communication validated
- Raspberry Pi → PC image streaming working
- OpenCV image display working

## Architecture

Windows PC
- Image reception
- Visualization
- Tracking (planned)
- IBVS (planned)

Raspberry Pi 5
- ROS 2 Jazzy
- Camera acquisition
- Safety authority
- ESP32 bridge

ESP32
- PID control
- Motor control
- Encoder feedback

## Development Progress

### Milestone 1
- UDP connectivity verified
- JPEG image streaming verified

### Next
- Packetized image protocol
- Dual camera support