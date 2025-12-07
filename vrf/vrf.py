from ecdsa import SigningKey, SECP256k1, VerifyingKey
import hashlib, binascii

class SimpleVRF:
    def __init__(self, sk=None):
        if sk:
            self.sk = SigningKey.from_string(binascii.unhexlify(sk), curve=SECP256k1)
        else:
            self.sk = SigningKey.generate(curve=SECP256k1)
        self.vk = self.sk.get_verifying_key()

    def prove(self, message: str):
        proof = self.sk.sign(message.encode())
        vrf_output = hashlib.sha256(proof).hexdigest()
        return vrf_output, proof.hex()

    def verify(self, message: str, proof_hex: str):
        try:
            proof = bytes.fromhex(proof_hex)
            self.vk.verify(proof, message.encode())
            vrf_output = hashlib.sha256(proof).hexdigest()
            return True, vrf_output
        except Exception:
            return False, None

    def get_public_key_hex(self):
        return self.vk.to_string().hex()

    def get_private_key_hex(self):
        return self.sk.to_string().hex()
