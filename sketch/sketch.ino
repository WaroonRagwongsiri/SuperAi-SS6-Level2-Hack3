#include <Arduino_RouterBridge.h>

void setup() {
  Serial.begin(115200);
  Bridge.begin();

  delay(5000);
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();

    if (msg.length() > 0) {
      Bridge.notify("serial_from_nano", msg);
    }
  }
}