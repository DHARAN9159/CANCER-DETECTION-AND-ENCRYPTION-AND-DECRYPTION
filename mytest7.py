# from cryptography.fernet import Fernet
#
# key = Fernet.generate_key()
print(key)
f = Fernet(key)
# print(key)
# encrypted_data = f.encrypt(b"This message is being encrypted and cannot be seen!")
# print(encrypted_data)
# # decrypted_data = f.decrypt(encrypted_data) # f is the variable that has the value of the key
# print(decrypted_data)
# f= Fernet(b'Usqlf7_J_qjGnEfjdWoMkNTaodAIiZNlYcgbaxSnriI=')
# encrypted = b'gAAAAABj-Y-LUBGUhAdN7SfrQ_zL2oeO2SETzSN2d_4LqfCRysZ_DVDWW4krpHSwbG1WcMah-AAxCJeiSnuo2hVr_zLXO1FP90D8Dzn5ASupO6P-5c5jreLqwWWCGgxLa3mjVGPe8c-G4qpQDvLCrGPq8vI10cQ-gg=='
# decrypted = f.decrypt(encrypted).decode("utf-8")
#
# print(decrypted)



