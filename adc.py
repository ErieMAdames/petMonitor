import Adafruit_ADS1x15

# Initialize the ADC
adc = Adafruit_ADS1x15.ADS1115()

# Gain configuration (1x gain for ±4.096V range)
GAIN = 1

# Read ADC values (A0, A1, A2, A3)
value_a0 = adc.read_adc(0, gain=GAIN)
value_a1 = 0#adc.read_adc(1, gain=GAIN)
value_a2 = 0#adc.read_adc(2, gain=GAIN)
value_a3 = 0#adc.read_adc(3, gain=GAIN)

# Print the results
print("A0: {}, A1: {}, A2: {}, A3: {}".format(value_a0, value_a1, value_a2, value_a3))
