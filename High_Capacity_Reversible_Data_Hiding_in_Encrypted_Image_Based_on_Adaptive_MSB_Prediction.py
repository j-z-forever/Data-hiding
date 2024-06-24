import numpy as np
import cv2
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import hashlib
import os


def key_to_int(key):
    return int(hashlib.sha256(key).hexdigest(), 16) % (2 ** 32)


def rc4_encrypt(data, key):
    algorithm = algorithms.ARC4(key)
    cipher = Cipher(algorithm, mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data) + encryptor.finalize()
    return encrypted_data


def block_level_encryption(image, key):
    key = key[:16]
    M, N = image.shape
    encrypted_image = np.zeros_like(image)

    for i in range(M):
        for j in range(N):
            pixel = np.array([image[i, j]], dtype=np.uint8).tobytes()
            encrypted_pixel = rc4_encrypt(pixel, key)
            encrypted_image[i, j] = np.frombuffer(encrypted_pixel, dtype=np.uint8)[0]

    return encrypted_image


def calculate_d(p, c):
    p_bin = format(p, '08b')
    c_bin = format(c, '08b')
    x = 0
    while x < 8 and p_bin[x] == c_bin[x]:
        x += 1
    return 8 - x


def adaptive_msb_prediction(block):
    P = block[0, 0]
    C1, C2, C3 = block[0, 1], block[1, 0], block[1, 1]
    d1 = calculate_d(P, C1)
    d2 = calculate_d(P, C2)
    d3 = calculate_d(P, C3)
    md = 8 - max(d1, d2, d3)
    if 2 <= md <= 8:
        e1 = C1 & ((1 << (8 - md)) - 1)
        e2 = C2 & ((1 << (8 - md)) - 1)
        e3 = C3 & ((1 << (8 - md)) - 1)
        return (P, md, e1, e2, e3), 3 * (md - 1)
    else:
        return None, 0


def generate_location_map(image):
    M, N = image.shape
    location_map = np.zeros((M // 2, N // 2), dtype=np.uint8)
    available_blocks = []
    unavailable_blocks = []

    for i in range(0, M, 2):
        for j in range(0, N, 2):
            block = image[i:i + 2, j:j + 2]
            _, emb_capacity = adaptive_msb_prediction(block)
            if emb_capacity > 0:
                location_map[i // 2, j // 2] = 1
                available_blocks.append(block)
            else:
                location_map[i // 2, j // 2] = 0
                unavailable_blocks.append(block)

    return location_map, available_blocks, unavailable_blocks


def rearrange_blocks(location_map, available_blocks, unavailable_blocks):
    M, N = location_map.shape
    rearranged_image = np.zeros((M * 2, N * 2), dtype=np.uint8)

    available_idx = 0
    unavailable_idx = len(unavailable_blocks) - 1

    for i in range(M):
        for j in range(N):
            if location_map[i, j] == 1:
                rearranged_image[i * 2:i * 2 + 2, j * 2:j * 2 + 2] = available_blocks[available_idx]
                available_idx += 1
            else:
                rearranged_image[i * 2:i * 2 + 2, j * 2:j * 2 + 2] = unavailable_blocks[unavailable_idx]
                unavailable_idx -= 1

    return rearranged_image


def encrypt_data(data, key):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()

    encrypted_data = iv + encryptor.update(padded_data) + encryptor.finalize()
    return encrypted_data


def embed_data(rearranged_image, encrypted_data, key, location_map):
    M, N = rearranged_image.shape
    embedded_image = np.copy(rearranged_image)

    np.random.seed(key_to_int(key))
    data_idx = 0
    total_bits_embedded = 0

    encrypted_data_bin = ''.join(format(byte, '08b') for byte in encrypted_data)
    print("Length of encrypted data (bits) before embedding:", len(encrypted_data_bin))

    if len(encrypted_data_bin) > M * N:
        raise ValueError("Encrypted data is too large to embed in the image")

    for i in range(M // 2):
        for j in range(N // 2):
            if location_map[i, j] == 1:
                block = rearranged_image[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
                prediction, emb_capacity = adaptive_msb_prediction(block)
                if prediction is not None and data_idx + emb_capacity <= len(encrypted_data_bin):
                    P, md, e1, e2, e3 = prediction
                    bits_to_embed = encrypted_data_bin[data_idx:data_idx + emb_capacity]
                    data_idx += emb_capacity

                    p_bin = format(P, '08b')
                    md_bin = format(md - 1, '03b')
                    e1_bin = format(e1, '0' + str(8 - md) + 'b')
                    e2_bin = format(e2, '0' + str(8 - md) + 'b')
                    e3_bin = format(e3, '0' + str(8 - md) + 'b')
                    combined_bits = p_bin + md_bin + e1_bin + e2_bin + e3_bin + bits_to_embed

                    if len(combined_bits) < 32:
                        combined_bits = combined_bits.ljust(32, '0')

                    new_p = int(combined_bits[:8], 2)
                    new_c1 = int(combined_bits[8:16], 2)
                    new_c2 = int(combined_bits[16:24], 2)
                    new_c3 = int(combined_bits[24:], 2)

                    embedded_image[i * 2, j * 2] = np.array(new_p).astype(np.uint8)
                    embedded_image[i * 2, j * 2 + 1] = np.array(new_c1).astype(np.uint8)
                    embedded_image[i * 2 + 1, j * 2] = np.array(new_c2).astype(np.uint8)
                    embedded_image[i * 2 + 1, j * 2 + 1] = np.array(new_c3).astype(np.uint8)
                    total_bits_embedded += len(bits_to_embed)

    print(f"Total bits embedded: {total_bits_embedded}")
    return embedded_image


def extract_data(embedded_image, location_map):
    M, N = embedded_image.shape
    extracted_data = []

    for i in range(M // 2):
        for j in range(N // 2):
            if location_map[i, j] == 1:
                block = embedded_image[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
                p_bin = format(block[0, 0], '08b')
                c1_bin = format(block[0, 1], '08b')
                c2_bin = format(block[1, 0], '08b')
                c3_bin = format(block[1, 1], '08b')

                combined_bits = p_bin + c1_bin + c2_bin + c3_bin
                md = int(combined_bits[8:11], 2) + 1
                if 2 <= md <= 8:
                    embedded_bits_bin = combined_bits[11 + 3 * (8 - md):]
                    extracted_bits = [int(bit) for bit in embedded_bits_bin[:3 * (md - 1)]]
                    extracted_data.extend(extracted_bits)

    extracted_data_str = ''.join(map(str, extracted_data))
    if len(extracted_data_str) % 8 != 0:
        extracted_data_str = extracted_data_str.ljust(len(extracted_data_str) + (8 - len(extracted_data_str) % 8), '0')
    extracted_bytes = int(extracted_data_str, 2).to_bytes((len(extracted_data_str) + 7) // 8, byteorder='big')
    print("Length of extracted data (bits) after extraction:", len(extracted_data_str))
    return extracted_bytes


def recover_image(embedded_image, location_map):
    M, N = embedded_image.shape
    recovered_image = np.copy(embedded_image)

    for i in range(M // 2):
        for j in range(N // 2):
            if location_map[i, j] == 1:
                block = embedded_image[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
                p_bin = format(block[0, 0], '08b')
                c1_bin = format(block[0, 1], '08b')
                c2_bin = format(block[1, 0], '08b')
                c3_bin = format(block[1, 1], '08b')

                combined_bits = p_bin + c1_bin + c2_bin + c3_bin
                md = int(combined_bits[8:11], 2) + 1
                if 2 <= md <= 8:
                    e1_bin = combined_bits[11:11 + (8 - md)]
                    e2_bin = combined_bits[11 + (8 - md):11 + 2 * (8 - md)]
                    e3_bin = combined_bits[11 + 2 * (8 - md):11 + 3 * (8 - md)]

                    e1 = int(e1_bin, 2)
                    e2 = int(e2_bin, 2)
                    e3 = int(e3_bin, 2)

                    original_c1_bin = p_bin[:md] + e1_bin
                    original_c2_bin = p_bin[:md] + e2_bin
                    original_c3_bin = p_bin[:md] + e3_bin

                    original_c1 = int(original_c1_bin, 2)
                    original_c2 = int(original_c2_bin, 2)
                    original_c3 = int(original_c3_bin, 2)

                    recovered_image[i * 2, j * 2] = block[0, 0]
                    recovered_image[i * 2, j * 2 + 1] = np.array(original_c1).astype(np.uint8)
                    recovered_image[i * 2 + 1, j * 2] = np.array(original_c2).astype(np.uint8)
                    recovered_image[i * 2 + 1, j * 2 + 1] = np.array(original_c3).astype(np.uint8)

    return recovered_image


def decrypt_data(encrypted_data, key):
    iv = encrypted_data[:16]
    encrypted_data = encrypted_data[16:]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()

    return data


image_path = 'D:\\programing\\python\\python-works\\experiment\\High_Capacity_Reversible_Data_Hiding_in_Encrypted_Image_Based_on_Adaptive_MSB_Prediction\\Peppers.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
block_key = os.urandom(16)
data_key = os.urandom(16)
data = np.random.randint(0, 2, size=10000, dtype=np.uint8).tolist()

encrypted_image = block_level_encryption(image, block_key)
location_map, available_blocks, unavailable_blocks = generate_location_map(encrypted_image)
rearranged_image = rearrange_blocks(location_map, available_blocks, unavailable_blocks)
encrypted_data = encrypt_data(bytes(data), data_key)
print("Length of encrypted data (bits):", len(encrypted_data) * 8)

embedded_image = embed_data(rearranged_image, encrypted_data, data_key, location_map)

# 提取嵌入的数据并解密
extracted_data = extract_data(embedded_image, location_map)
print("Length of extracted data (bits):", len(extracted_data) * 8)

decrypted_data = decrypt_data(extracted_data, data_key)
decrypted_data_bin = ''.join(format(byte, '08b') for byte in decrypted_data)

recovered_image = recover_image(embedded_image, location_map)

print("Original Image:\n", image)
print("Encrypted Image:\n", encrypted_image)
print("Rearranged Image:\n", rearranged_image)
print("Embedded Image:\n", embedded_image)
print("Extracted Data (Binary):\n", decrypted_data_bin)
print("Recovered Image:\n", recovered_image)

if bytes(data) == decrypted_data:
    print("Data embedded and extracted successfully match!")
else:
    print("Data mismatch! There is an issue with embedding or extraction.")
