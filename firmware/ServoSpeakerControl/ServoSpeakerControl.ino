#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <ESP32Servo.h>
#include <ESP_I2S.h>
#include "secrets.h"

// network
const char* HOSTNAME = "servo-speaker";  // reachable as servo-speaker.local
const int HTTP_PORT = 80;                // servo requests
const int AUDIO_PORT = 81;               // raw audio stream

// servo
const int SERVO_PIN = 14;
const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;

// speaker (I2S amplifier)
const int I2S_BCLK = 26;
const int I2S_LRC = 25;
const int I2S_DIN = 27;
const int SAMPLE_RATE = 16000;           // must match SPEAKER_SAMPLE_RATE in the Python config
const int VOLUME_PERCENT = 60;           // 100 = full volume
const int RX_BYTES = 1024;               // network bytes read per chunk
const unsigned long AUDIO_TIMEOUT_MS = 3000;  // stop if the sender goes quiet

Servo servo;
I2SClass i2s;
WebServer httpServer(HTTP_PORT);
WiFiServer audioServer(AUDIO_PORT);

uint8_t rxBuffer[RX_BYTES];   // raw 16-bit mono bytes from the network
int16_t outBuffer[RX_BYTES];  // the same samples as stereo (each sample twice)

// POST /servo?angle=NN -> move the servo
void handleServo() {
    if (!httpServer.hasArg("angle")) {
        httpServer.send(400, "text/plain", "missing angle");
        return;
    }

    int angle = constrain(httpServer.arg("angle").toInt(), SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
    servo.write(angle);
    Serial.printf("Servo -> %d\n", angle);
    httpServer.send(200, "text/plain", String(angle));
}

// play raw 16-bit mono PCM from a client until it disconnects
void playAudio(WiFiClient& client) {
    Serial.println("Playing audio");
    size_t held = 0;  // leftover byte when a read ends in the middle of a sample
    unsigned long lastData = millis();

    while (client.connected() || client.available()) {
        int n = client.read(rxBuffer + held, RX_BYTES - held);
        if (n <= 0) {
            if (millis() - lastData > AUDIO_TIMEOUT_MS) break;
            delay(1);
            continue;
        }
        lastData = millis();

        size_t total = held + n;
        size_t samples = total / 2;
        for (size_t i = 0; i < samples; i++) {
            int32_t s = (int16_t)(rxBuffer[2 * i] | (rxBuffer[2 * i + 1] << 8));
            s = s * VOLUME_PERCENT / 100;
            outBuffer[2 * i] = s;      // left
            outBuffer[2 * i + 1] = s;  // right
        }
        i2s.write((uint8_t*)outBuffer, samples * 4);  // blocks until played, which paces the stream

        held = total % 2;
        if (held) rxBuffer[0] = rxBuffer[total - 1];
    }

    client.stop();
    Serial.println("Audio done");
}

void setup() {
    Serial.begin(115200);

    // servo
    servo.setPeriodHertz(50);
    servo.attach(SERVO_PIN);
    servo.write(90);

    // speaker
    i2s.setPins(I2S_BCLK, I2S_LRC, I2S_DIN);
    if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO)) {
        Serial.println("Failed to start I2S");
    }

    // wifi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\nConnected. IP: %s\n", WiFi.localIP().toString().c_str());
    MDNS.begin(HOSTNAME);

    // servers
    httpServer.on("/servo", HTTP_POST, handleServo);
    httpServer.begin();
    audioServer.begin();
    Serial.println("Servers started");
}

void loop() {
    httpServer.handleClient();

    WiFiClient client = audioServer.accept();
    if (client) playAudio(client);
}
