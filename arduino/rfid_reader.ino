#include <WiFi.h>
#include <SPI.h>
#include <MFRC522.h>
#include <WebServer.h>
#define SS_PIN  21
#define RST_PIN 22
MFRC522 rfid(SS_PIN, RST_PIN);
const char* ssid = "AIE_509_2.4G";
const char* password = "addinedu_class1";
String currentUID = "";
WebServer server(80);
void setup() {
  Serial.begin(115200);
  SPI.begin();
  rfid.PCD_Init();
  Serial.println(":하늘색_다이아몬드: RFID Ready");
  // Wi-Fi 연결
  WiFi.begin(ssid, password);
  Serial.print(":신호_강도: WiFi connecting...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n:흰색_확인_표시: WiFi connected. IP: " + WiFi.localIP().toString());
  // UID 요청 처리 핸들러 등록
  server.on("/uid", HTTP_GET, []() {
  server.send(200, "application/json", "{\"uid\":\"" + currentUID + "\"}");
  if (currentUID != "") {
    Serial.println(":보낼_편지함_트레이: UID 응답 후 초기화: " + currentUID);
    currentUID = "";
  }
});
  server.begin();
  Serial.println(":자오선이_있는_지구: Web server started");
}
void loop() {
  server.handleClient();
  if (!rfid.PICC_IsNewCardPresent() || !rfid.PICC_ReadCardSerial()) return;
  String uidStr = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    uidStr += (rfid.uid.uidByte[i] < 0x10 ? "0" : "");
    uidStr += String(rfid.uid.uidByte[i], HEX);
  }
  uidStr.toUpperCase();
  if (uidStr != currentUID && uidStr != "") {
    currentUID = uidStr;
    Serial.println(":흰색_확인_표시: New UID: " + currentUID);
  }
  delay(1000);
  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
}