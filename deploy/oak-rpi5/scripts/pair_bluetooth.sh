#!/usr/bin/env bash
# Helper script to pair Bluetooth headphones on Raspberry Pi 5
set -euo pipefail

echo "=== Bluetooth Headphone Pairing ==="
echo ""
echo "Put your headphones in pairing mode, then press Enter..."
read -r

echo ""
echo "Scanning for Bluetooth devices (30 seconds)..."
echo ""

# Start scanning
bluetoothctl --timeout 30 scan on &
SCAN_PID=$!
sleep 30
kill "$SCAN_PID" 2>/dev/null || true

echo ""
echo "Available audio devices:"
echo ""
bluetoothctl devices | while read -r line; do
    echo "  $line"
done

echo ""
echo "Enter the MAC address of your headphones (e.g., AA:BB:CC:DD:EE:FF):"
read -r MAC

if [ -z "$MAC" ]; then
    echo "No MAC address entered. Exiting."
    exit 1
fi

echo ""
echo "Pairing with $MAC..."
bluetoothctl pair "$MAC"

echo "Trusting $MAC..."
bluetoothctl trust "$MAC"

echo "Connecting to $MAC..."
bluetoothctl connect "$MAC"

echo ""
echo "Setting as default audio sink..."
# Wait for PulseAudio to detect the device
sleep 3

# Find the Bluetooth sink
BT_SINK=$(pactl list short sinks | grep -i "bluez" | head -1 | awk '{print $2}')
if [ -n "$BT_SINK" ]; then
    pactl set-default-sink "$BT_SINK"
    echo "Default sink set to: $BT_SINK"
else
    echo "WARNING: Bluetooth audio sink not found. Try:"
    echo "  pactl list short sinks"
    echo "  pactl set-default-sink <sink-name>"
fi

# Find the Bluetooth source (microphone)
BT_SOURCE=$(pactl list short sources | grep -i "bluez" | head -1 | awk '{print $2}')
if [ -n "$BT_SOURCE" ]; then
    pactl set-default-source "$BT_SOURCE"
    echo "Default source set to: $BT_SOURCE"
else
    echo "WARNING: Bluetooth audio source not found."
    echo "Your headphones may not support HFP (hands-free profile)."
    echo "A USB microphone can be used as an alternative."
fi

echo ""
echo "=== Pairing complete! ==="
echo "Test audio: aplay /usr/share/sounds/alsa/Front_Center.wav"
