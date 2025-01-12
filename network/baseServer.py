import shutil
import rsa
import hashlib
class BaseServer:
    """
    基础服务器类，提供文件传输、密钥生成、文件加密和文件哈希生成功能。
    """

    def __init__(self):
        """
        初始化BaseServer实例，生成RSA密钥对。
        """
        self.rsa_public_key, self.rsa_private_key = self.generate_key()

    def send_file(self, root_path, target_path):
        """
        将文件从源路径复制到目标路径。

        :param root_path: 源文件路径
        :param target_path: 目标文件路径
        """
        shutil.copy(root_path, target_path)

    def generate_key(self, n_bits: int = 2048):
        """
        生成RSA密钥对。

        :param n_bits: 密钥长度，默认为2048位
        :return: 生成的RSA公钥和私钥
        """
        return rsa.newkeys(n_bits)

    def encrypt_file(self, data, file_path):
        """
        使用RSA公钥加密文件内容。

        :param data: 要加密的数据
        :param file_path: 文件路径
        :return: 加密后的数据
        """
        public_key = self.rsa_public_key
        with open(file_path, 'rb') as f:
            data = f.read()
        encrypted_data = rsa.encrypt(data, public_key)
        return encrypted_data

    def generate_file_hash(self, file_path):
        """
        生成文件的MD5哈希值。

        :param file_path: 文件路径
        :return: 文件的MD5哈希值
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()