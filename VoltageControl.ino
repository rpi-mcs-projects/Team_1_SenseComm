/*
 * Voltage Control Example
 * 
 * This sketch demonstrates how to change the effective voltage on a PWM pin
 * from a starting value (V1) to an ending value (V2).
 * 
 * NOTE: On most standard Arduinos (Uno, Nano, Mega), 'analogWrite' produces 
 * a PWM (Pulse Width Modulation) signal, not a true analog voltage. 
 * To get a steady DC voltage, you would need to add a Low-Pass Filter (RC circuit)
 * to the output pin.
 * 
 * If you are using a board with a DAC (like Arduino Due, Zero, or ESP32), 
 * you can output true analog voltages on specific DAC pins.
 */

// Define the output pin (Must be a PWM-capable pin, marked with ~ on Uno)
const int outputPin = 9; 

// Define your target voltages (assuming a 5V logic level board)
// If using a 3.3V board, change the reference in the calculation below.
float V1 = 1.0; // Starting Voltage
float V2 = 3.5; // Ending Voltage

void setup() {
  // Initialize the pin as an output
  pinMode(outputPin, OUTPUT);
  
  // Initialize serial communication for debugging
  Serial.begin(9600);
  Serial.println("Voltage Control Demo Started");
}

void loop() {
  // 1. Set output to V1
  Serial.print("Outputting V1: ");
  Serial.print(V1);
  Serial.println(" Volts");
  setVoltage(V1);
  
  delay(3000); // Wait for 3 seconds

  // 2. Change output to V2
  Serial.print("Changing to V2: ");
  Serial.print(V2);
  Serial.println(" Volts");
  setVoltage(V2);
  
  delay(3000); // Wait for 3 seconds
  
  // Optional: Smooth transition (Ramp) from V2 back to V1
  Serial.println("Ramping back to V1...");
  rampVoltage(V2, V1, 2000); // Ramp over 2000 milliseconds
}

// Function to set the voltage immediately
void setVoltage(float voltage) {
  // Constrain the value to the board's range (0V to 5V)
  voltage = constrain(voltage, 0.0, 5.0);
  
  // Convert voltage to PWM duty cycle (0 - 255)
  // Formula: (Voltage / MaxVoltage) * MaxPWM
  int pwmValue = (voltage / 5.0) * 255;
  
  // Write the value to the pin
  analogWrite(outputPin, pwmValue);
}

// Function to smoothly ramp voltage from start to end over a duration
void rampVoltage(float startVolts, float endVolts, unsigned long durationMillis) {
  unsigned long startTime = millis();
  
  while (millis() - startTime <= durationMillis) {
    float progress = (float)(millis() - startTime) / durationMillis;
    float currentVolts = startVolts + (endVolts - startVolts) * progress;
    setVoltage(currentVolts);
    delay(10); // Small delay for stability
  }
  // Ensure we land exactly on the target
  setVoltage(endVolts);
}
