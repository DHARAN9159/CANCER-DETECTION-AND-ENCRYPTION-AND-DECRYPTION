# def getImageMatrix(imageName):
#     print("hello")
#     im = Image.open(imageName)
#
#     pix = im.load()
#     color = 1
#     if type(pix[0, 0]) == int:
#         color = 0
#     image_size = im.size
#     image_matrix = []
#     for width in range(int(image_size[0])):
#         row = []
#         for height in range(int(image_size[1])):
#             row.append((pix[width, height]))
#         image_matrix.append(row)
#     return image_matrix, image_size[0], image_size[1], color
#
#
# def LogisticEncryption(imageName, key):
#     N = 256
#     print("hello1")
#     key_list = [ord(x) for x in key]
#     G = [key_list[0:4], key_list[4:8], key_list[8:12]]
#     g = []
#     R = 1
#     for i in range(1, 4):
#         s = 0
#         for j in range(1, 5):
#             s += G[i - 1][j - 1] * (10 ** (-j))
#         g.append(s)
#         R = (R * s) % 1
#
#     L = (R + key_list[12] / 256) % 1
#     S_x = round(((g[0] + g[1] + g[2]) * (10 ** 4) + L * (10 ** 4)) % 256)
#     V1 = sum(key_list)
#     V2 = key_list[0]
#     for i in range(1, 13):
#         V2 = V2 ^ key_list[i]
#     V = V2 / V1
#
#     L_y = (V + key_list[12] / 256) % 1
#     S_y = round((V + V2 + L_y * 10 ** 4) % 256)
#     C1_0 = S_x
#     C2_0 = S_y
#     C = round((L * L_y * 10 ** 4) % 256)
#     C_r = round((L * L_y * 10 ** 4) % 256)
#     C_g = round((L * L_y * 10 ** 4) % 256)
#     C_b = round((L * L_y * 10 ** 4) % 256)
#     x = 4 * (S_x) * (1 - S_x)
#     y = 4 * (S_y) * (1 - S_y)
#
#     imageMatrix, dimensionX, dimensionY, color = getImageMatrix(imageName)
#     LogisticEncryptionIm = []
#     for i in range(dimensionX):
#         row = []
#         for j in range(dimensionY):
#             while x < 0.8 and x > 0.2:
#                 x = 4 * x * (1 - x)
#             while y < 0.8 and y > 0.2:
#                 y = 4 * y * (1 - y)
#             x_round = round((x * (10 ** 4)) % 256)
#             y_round = round((y * (10 ** 4)) % 256)
#             C1 = x_round ^ ((key_list[0] + x_round) % N) ^ ((C1_0 + key_list[1]) % N)
#             C2 = x_round ^ ((key_list[2] + y_round) % N) ^ ((C2_0 + key_list[3]) % N)
#             if color:
#                 C_r = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
#                         (key_list[6] + imageMatrix[i][j][0]) % N) ^ ((C_r + key_list[7]) % N)
#                 C_g = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
#                         (key_list[6] + imageMatrix[i][j][1]) % N) ^ ((C_g + key_list[7]) % N)
#                 C_b = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
#                         (key_list[6] + imageMatrix[i][j][2]) % N) ^ ((C_b + key_list[7]) % N)
#                 row.append((C_r, C_g, C_b))
#                 C = C_r
#
#             else:
#                 C = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
#                         (key_list[6] + imageMatrix[i][j]) % N) ^ (
#                             (C + key_list[7]) % N)
#                 row.append(C)
#
#             x = (x + C / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
#             y = (x + C / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
#             for ki in range(12):
#                 key_list[ki] = (key_list[ki] + key_list[12]) % 256
#                 key_list[12] = key_list[12] ^ key_list[ki]
#         LogisticEncryptionIm.append(row)
#
#     im = Image.new("L", (dimensionX, dimensionY))
#     if color:
#         im = Image.new("RGB", (dimensionX, dimensionY))
#     else:
#         im = Image.new("L", (dimensionX, dimensionY))  # L is for Black and white pixels
#
#     pix = im.load()
#     for x in range(dimensionX):
#         for y in range(dimensionY):
#             pix[x, y] = LogisticEncryptionIm[x][y]
#             print(pix[x, y])
#     im.save(imageName.split('.')[0] + "_encrypted.png", "PNG")
#
#     # xx1=pix.open()
#     # print(xx1)
#
#     r = data.id
#     st = upload.objects.filter(id=r).update(Encryptedimage=im1)
#
#
# try:
#
#     LogisticEncryption(path, "asdsfdsvghdfbg")
#     print(path)
# except:
#     LogisticEncryption(path, "asdsfdsvghdfbg")
#
# messages.info(request, "Successfully Encrypted ")
# return redirect("/encrypt/")
