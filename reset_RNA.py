# reset_models.py
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

def reset_tensorflow_models():
    """Reseta completamente todos os modelos e estados do TensorFlow/Keras"""
    try:
        # Encerra a sessão atual do TensorFlow
        K.clear_session()
        
        # Libera memória GPU/CPU
        if tf.config.list_physical_devices('GPU'):
            tf.compat.v1.reset_default_graph()
        
        # Coletor de lixo para limpeza profunda
        gc.collect()
        
        print(" Redes neurais e sessão do TensorFlow resetadas com sucesso!")
        return True
    except Exception as e:
        print(f" Falha ao resetar modelos: {str(e)}")
        return False

if __name__ == "__main__":
    reset_tensorflow_models()
# reset_models.py
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

def reset_tensorflow_models():
    """Reseta completamente todos os modelos e estados do TensorFlow/Keras"""
    try:
        # Encerra a sessão atual do TensorFlow
        K.clear_session()
        
        # Libera memória GPU/CPU
        if tf.config.list_physical_devices('GPU'):
            tf.compat.v1.reset_default_graph()
        
        # Coletor de lixo para limpeza profunda
        gc.collect()
        
        print(" Redes neurais e sessão do TensorFlow resetadas com sucesso!")
        return True
    except Exception as e:
        print(f" Falha ao resetar modelos: {str(e)}")
        return False

if __name__ == "__main__":
    reset_tensorflow_models()