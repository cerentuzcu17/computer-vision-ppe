import cv2
import numpy as np
from cryptography.fernet import Fernet

class ImageEncryptor:
    def __init__(self, key: bytes = None):
        """
        Initializes the encryptor. 
        In production, this key should be loaded securely from a .env file.
        """
        self.key = key if key else Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt_frame(self, frame_np: np.ndarray) -> bytes:
        """
        Converts an OpenCV frame (numpy array) to raw JPEG bytes in memory and encrypts it.
        """
        success, encoded_image = cv2.imencode('.jpg', frame_np)
        if not success:
            raise ValueError("Failed to encode frame to byte format.")
            
        raw_bytes = encoded_image.tobytes()
        return self.cipher.encrypt(raw_bytes)

    def decrypt_frame(self, encrypted_bytes: bytes) -> np.ndarray:
        """
        Decrypts the raw encrypted bytes and decodes them back into an OpenCV frame (numpy array).
        """
        decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
        nparr = np.frombuffer(decrypted_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
