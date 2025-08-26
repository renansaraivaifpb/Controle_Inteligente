import re
import matplotlib.pyplot as plt
import numpy as np

# O log de treinamento que você forneceu, armazenado como uma string de múltiplas linhas.
training_log = """
Epoch 1/150
12/12 - 2s - loss: 0.9254 - accuracy: 0.5731 - val_loss: 0.6488 - val_accuracy: 0.7514 - 2s/epoch - 136ms/step
Epoch 2/150
12/12 - 0s - loss: 0.7021 - accuracy: 0.7253 - val_loss: 0.6290 - val_accuracy: 0.6757 - 83ms/epoch - 7ms/step
Epoch 3/150
12/12 - 0s - loss: 0.6437 - accuracy: 0.7226 - val_loss: 0.5246 - val_accuracy: 0.8000 - 81ms/epoch - 7ms/step
Epoch 4/150
12/12 - 0s - loss: 0.5932 - accuracy: 0.7524 - val_loss: 0.5100 - val_accuracy: 0.7703 - 81ms/epoch - 7ms/step
Epoch 5/150
12/12 - 0s - loss: 0.5541 - accuracy: 0.7896 - val_loss: 0.4806 - val_accuracy: 0.8054 - 81ms/epoch - 7ms/step
Epoch 6/150
12/12 - 0s - loss: 0.4834 - accuracy: 0.8119 - val_loss: 0.4209 - val_accuracy: 0.8432 - 81ms/epoch - 7ms/step
Epoch 7/150
12/12 - 0s - loss: 0.4373 - accuracy: 0.8302 - val_loss: 0.3979 - val_accuracy: 0.8649 - 82ms/epoch - 7ms/step
Epoch 8/150
12/12 - 0s - loss: 0.4195 - accuracy: 0.8356 - val_loss: 0.4061 - val_accuracy: 0.8405 - 51ms/epoch - 4ms/step
Epoch 9/150
12/12 - 0s - loss: 0.3634 - accuracy: 0.8708 - val_loss: 0.4205 - val_accuracy: 0.8622 - 50ms/epoch - 4ms/step
Epoch 10/150
12/12 - 0s - loss: 0.3217 - accuracy: 0.8755 - val_loss: 0.4018 - val_accuracy: 0.8649 - 51ms/epoch - 4ms/step
Epoch 11/150
12/12 - 0s - loss: 0.2850 - accuracy: 0.8863 - val_loss: 0.4021 - val_accuracy: 0.8676 - 51ms/epoch - 4ms/step
Epoch 12/150
12/12 - 0s - loss: 0.3023 - accuracy: 0.8904 - val_loss: 0.3676 - val_accuracy: 0.8757 - 80ms/epoch - 7ms/step
Epoch 13/150
12/12 - 0s - loss: 0.2800 - accuracy: 0.8924 - val_loss: 0.3903 - val_accuracy: 0.8568 - 51ms/epoch - 4ms/step
Epoch 14/150
12/12 - 0s - loss: 0.2543 - accuracy: 0.8992 - val_loss: 0.3882 - val_accuracy: 0.8757 - 50ms/epoch - 4ms/step
Epoch 15/150
12/12 - 0s - loss: 0.2263 - accuracy: 0.9188 - val_loss: 0.3800 - val_accuracy: 0.8838 - 49ms/epoch - 4ms/step
Epoch 16/150
12/12 - 0s - loss: 0.2164 - accuracy: 0.9195 - val_loss: 0.4122 - val_accuracy: 0.8811 - 50ms/epoch - 4ms/step
Epoch 17/150
12/12 - 0s - loss: 0.2022 - accuracy: 0.9290 - val_loss: 0.3959 - val_accuracy: 0.8892 - 50ms/epoch - 4ms/step
Epoch 18/150
12/12 - 0s - loss: 0.1773 - accuracy: 0.9425 - val_loss: 0.4067 - val_accuracy: 0.8703 - 50ms/epoch - 4ms/step
Epoch 19/150
12/12 - 0s - loss: 0.1604 - accuracy: 0.9479 - val_loss: 0.4251 - val_accuracy: 0.8730 - 51ms/epoch - 4ms/step
Epoch 20/150
12/12 - 0s - loss: 0.1465 - accuracy: 0.9479 - val_loss: 0.4540 - val_accuracy: 0.8838 - 50ms/epoch - 4ms/step
Epoch 21/150
12/12 - 0s - loss: 0.1176 - accuracy: 0.9553 - val_loss: 0.4504 - val_accuracy: 0.8784 - 49ms/epoch - 4ms/step
Epoch 22/150
12/12 - 0s - loss: 0.1315 - accuracy: 0.9520 - val_loss: 0.4513 - val_accuracy: 0.8838 - 49ms/epoch - 4ms/step
Epoch 23/150
12/12 - 0s - loss: 0.1178 - accuracy: 0.9648 - val_loss: 0.4571 - val_accuracy: 0.8811 - 49ms/epoch - 4ms/step
Epoch 24/150
12/12 - 0s - loss: 0.1261 - accuracy: 0.9553 - val_loss: 0.4606 - val_accuracy: 0.8757 - 50ms/epoch - 4ms/step
Epoch 25/150
12/12 - 0s - loss: 0.1247 - accuracy: 0.9547 - val_loss: 0.4604 - val_accuracy: 0.8865 - 50ms/epoch - 4ms/step
Epoch 26/150
12/12 - 0s - loss: 0.1007 - accuracy: 0.9675 - val_loss: 0.4434 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 27/150
12/12 - 0s - loss: 0.0971 - accuracy: 0.9668 - val_loss: 0.4555 - val_accuracy: 0.9027 - 51ms/epoch - 4ms/step
Epoch 28/150
12/12 - 0s - loss: 0.1065 - accuracy: 0.9648 - val_loss: 0.4692 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 29/150
12/12 - 0s - loss: 0.0779 - accuracy: 0.9783 - val_loss: 0.4523 - val_accuracy: 0.8892 - 50ms/epoch - 4ms/step
Epoch 30/150
12/12 - 0s - loss: 0.0780 - accuracy: 0.9736 - val_loss: 0.4892 - val_accuracy: 0.8946 - 49ms/epoch - 4ms/step
Epoch 31/150
12/12 - 0s - loss: 0.0953 - accuracy: 0.9635 - val_loss: 0.4682 - val_accuracy: 0.8919 - 49ms/epoch - 4ms/step
Epoch 32/150
12/12 - 0s - loss: 0.0915 - accuracy: 0.9662 - val_loss: 0.4992 - val_accuracy: 0.8973 - 50ms/epoch - 4ms/step
Epoch 33/150
12/12 - 0s - loss: 0.0728 - accuracy: 0.9750 - val_loss: 0.5382 - val_accuracy: 0.8865 - 49ms/epoch - 4ms/step
Epoch 34/150
12/12 - 0s - loss: 0.0753 - accuracy: 0.9743 - val_loss: 0.5249 - val_accuracy: 0.8892 - 50ms/epoch - 4ms/step
Epoch 35/150
12/12 - 0s - loss: 0.0564 - accuracy: 0.9838 - val_loss: 0.5057 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 36/150
12/12 - 0s - loss: 0.0558 - accuracy: 0.9851 - val_loss: 0.5300 - val_accuracy: 0.8838 - 50ms/epoch - 4ms/step
Epoch 37/150
12/12 - 0s - loss: 0.0585 - accuracy: 0.9824 - val_loss: 0.5570 - val_accuracy: 0.8757 - 50ms/epoch - 4ms/step
Epoch 38/150
12/12 - 0s - loss: 0.0524 - accuracy: 0.9811 - val_loss: 0.5435 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 39/150
12/12 - 0s - loss: 0.0500 - accuracy: 0.9838 - val_loss: 0.4985 - val_accuracy: 0.9000 - 63ms/epoch - 5ms/step
Epoch 40/150
12/12 - 0s - loss: 0.0534 - accuracy: 0.9824 - val_loss: 0.4839 - val_accuracy: 0.9054 - 49ms/epoch - 4ms/step
Epoch 41/150
12/12 - 0s - loss: 0.0387 - accuracy: 0.9899 - val_loss: 0.5357 - val_accuracy: 0.8892 - 49ms/epoch - 4ms/step
Epoch 42/150
12/12 - 0s - loss: 0.0464 - accuracy: 0.9858 - val_loss: 0.5513 - val_accuracy: 0.8973 - 49ms/epoch - 4ms/step
Epoch 43/150
12/12 - 0s - loss: 0.0478 - accuracy: 0.9865 - val_loss: 0.4973 - val_accuracy: 0.9027 - 50ms/epoch - 4ms/step
Epoch 44/150
12/12 - 0s - loss: 0.0482 - accuracy: 0.9885 - val_loss: 0.5010 - val_accuracy: 0.9000 - 52ms/epoch - 4ms/step
Epoch 45/150
12/12 - 0s - loss: 0.0384 - accuracy: 0.9871 - val_loss: 0.5248 - val_accuracy: 0.9000 - 55ms/epoch - 5ms/step
Epoch 46/150
12/12 - 0s - loss: 0.0314 - accuracy: 0.9905 - val_loss: 0.5471 - val_accuracy: 0.8919 - 51ms/epoch - 4ms/step
Epoch 47/150
12/12 - 0s - loss: 0.0315 - accuracy: 0.9919 - val_loss: 0.6185 - val_accuracy: 0.8919 - 51ms/epoch - 4ms/step
Epoch 48/150
12/12 - 0s - loss: 0.0410 - accuracy: 0.9912 - val_loss: 0.5728 - val_accuracy: 0.8865 - 55ms/epoch - 5ms/step
Epoch 49/150
12/12 - 0s - loss: 0.0294 - accuracy: 0.9946 - val_loss: 0.6087 - val_accuracy: 0.8865 - 50ms/epoch - 4ms/step
Epoch 50/150
12/12 - 0s - loss: 0.0363 - accuracy: 0.9919 - val_loss: 0.5718 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 51/150
12/12 - 0s - loss: 0.0255 - accuracy: 0.9926 - val_loss: 0.6283 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 52/150
12/12 - 0s - loss: 0.0411 - accuracy: 0.9885 - val_loss: 0.6226 - val_accuracy: 0.8919 - 49ms/epoch - 4ms/step
Epoch 53/150
12/12 - 0s - loss: 0.0446 - accuracy: 0.9817 - val_loss: 0.5918 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 54/150
12/12 - 0s - loss: 0.0465 - accuracy: 0.9858 - val_loss: 0.6043 - val_accuracy: 0.8892 - 50ms/epoch - 4ms/step
Epoch 55/150
12/12 - 0s - loss: 0.0718 - accuracy: 0.9783 - val_loss: 0.5863 - val_accuracy: 0.8973 - 50ms/epoch - 4ms/step
Epoch 56/150
12/12 - 0s - loss: 0.0459 - accuracy: 0.9878 - val_loss: 0.5789 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 57/150
12/12 - 0s - loss: 0.0316 - accuracy: 0.9919 - val_loss: 0.6931 - val_accuracy: 0.8892 - 51ms/epoch - 4ms/step
Epoch 58/150
12/12 - 0s - loss: 0.0248 - accuracy: 0.9939 - val_loss: 0.6837 - val_accuracy: 0.8838 - 50ms/epoch - 4ms/step
Epoch 59/150
12/12 - 0s - loss: 0.0282 - accuracy: 0.9926 - val_loss: 0.6673 - val_accuracy: 0.8946 - 49ms/epoch - 4ms/step
Epoch 60/150
12/12 - 0s - loss: 0.0303 - accuracy: 0.9892 - val_loss: 0.7699 - val_accuracy: 0.8811 - 51ms/epoch - 4ms/step
Epoch 61/150
12/12 - 0s - loss: 0.0385 - accuracy: 0.9905 - val_loss: 0.6727 - val_accuracy: 0.8892 - 50ms/epoch - 4ms/step
Epoch 62/150
12/12 - 0s - loss: 0.0295 - accuracy: 0.9939 - val_loss: 0.5896 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 63/150
12/12 - 0s - loss: 0.0267 - accuracy: 0.9926 - val_loss: 0.6648 - val_accuracy: 0.8838 - 50ms/epoch - 4ms/step
Epoch 64/150
12/12 - 0s - loss: 0.0364 - accuracy: 0.9919 - val_loss: 0.6164 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 65/150
12/12 - 0s - loss: 0.0375 - accuracy: 0.9899 - val_loss: 0.6155 - val_accuracy: 0.8946 - 49ms/epoch - 4ms/step
Epoch 66/150
12/12 - 0s - loss: 0.0233 - accuracy: 0.9953 - val_loss: 0.6571 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 67/150
12/12 - 0s - loss: 0.0326 - accuracy: 0.9885 - val_loss: 0.6402 - val_accuracy: 0.9081 - 51ms/epoch - 4ms/step
Epoch 68/150
12/12 - 0s - loss: 0.0387 - accuracy: 0.9939 - val_loss: 0.6466 - val_accuracy: 0.8946 - 52ms/epoch - 4ms/step
Epoch 69/150
12/12 - 0s - loss: 0.0286 - accuracy: 0.9919 - val_loss: 0.6663 - val_accuracy: 0.8730 - 50ms/epoch - 4ms/step
Epoch 70/150
12/12 - 0s - loss: 0.0328 - accuracy: 0.9919 - val_loss: 0.6431 - val_accuracy: 0.9000 - 50ms/epoch - 4ms/step
Epoch 71/150
12/12 - 0s - loss: 0.0333 - accuracy: 0.9926 - val_loss: 0.5627 - val_accuracy: 0.9081 - 49ms/epoch - 4ms/step
Epoch 72/150
12/12 - 0s - loss: 0.0337 - accuracy: 0.9905 - val_loss: 0.7053 - val_accuracy: 0.8892 - 49ms/epoch - 4ms/step
Epoch 73/150
12/12 - 0s - loss: 0.0307 - accuracy: 0.9905 - val_loss: 0.7316 - val_accuracy: 0.8757 - 50ms/epoch - 4ms/step
Epoch 74/150
12/12 - 0s - loss: 0.0303 - accuracy: 0.9892 - val_loss: 0.7139 - val_accuracy: 0.8811 - 50ms/epoch - 4ms/step
Epoch 75/150
12/12 - 0s - loss: 0.0300 - accuracy: 0.9939 - val_loss: 0.7843 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 76/150
12/12 - 0s - loss: 0.0261 - accuracy: 0.9919 - val_loss: 0.7762 - val_accuracy: 0.8973 - 50ms/epoch - 4ms/step
Epoch 77/150
12/12 - 0s - loss: 0.0231 - accuracy: 0.9932 - val_loss: 0.7288 - val_accuracy: 0.8946 - 49ms/epoch - 4ms/step
Epoch 78/150
12/12 - 0s - loss: 0.0220 - accuracy: 0.9932 - val_loss: 0.7614 - val_accuracy: 0.9000 - 49ms/epoch - 4ms/step
Epoch 79/150
12/12 - 0s - loss: 0.0225 - accuracy: 0.9959 - val_loss: 0.7652 - val_accuracy: 0.8973 - 50ms/epoch - 4ms/step
Epoch 80/150
12/12 - 0s - loss: 0.0179 - accuracy: 0.9966 - val_loss: 0.7565 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 81/150
12/12 - 0s - loss: 0.0181 - accuracy: 0.9946 - val_loss: 0.7637 - val_accuracy: 0.8892 - 54ms/epoch - 5ms/step
Epoch 82/150
12/12 - 0s - loss: 0.0274 - accuracy: 0.9932 - val_loss: 0.7244 - val_accuracy: 0.8946 - 51ms/epoch - 4ms/step
Epoch 83/150
12/12 - 0s - loss: 0.0186 - accuracy: 0.9959 - val_loss: 0.7653 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 84/150
12/12 - 0s - loss: 0.0232 - accuracy: 0.9939 - val_loss: 0.8045 - val_accuracy: 0.8973 - 50ms/epoch - 4ms/step
Epoch 85/150
12/12 - 0s - loss: 0.0259 - accuracy: 0.9932 - val_loss: 0.7468 - val_accuracy: 0.8892 - 49ms/epoch - 4ms/step
Epoch 86/150
12/12 - 0s - loss: 0.0282 - accuracy: 0.9939 - val_loss: 0.7963 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 87/150
12/12 - 0s - loss: 0.0505 - accuracy: 0.9851 - val_loss: 0.8108 - val_accuracy: 0.8865 - 50ms/epoch - 4ms/step
Epoch 88/150
12/12 - 0s - loss: 0.0402 - accuracy: 0.9885 - val_loss: 0.7484 - val_accuracy: 0.8973 - 50ms/epoch - 4ms/step
Epoch 89/150
12/12 - 0s - loss: 0.0271 - accuracy: 0.9953 - val_loss: 0.7774 - val_accuracy: 0.8946 - 51ms/epoch - 4ms/step
Epoch 90/150
12/12 - 0s - loss: 0.0297 - accuracy: 0.9932 - val_loss: 0.7083 - val_accuracy: 0.9000 - 52ms/epoch - 4ms/step
Epoch 91/150
12/12 - 0s - loss: 0.0357 - accuracy: 0.9899 - val_loss: 0.7508 - val_accuracy: 0.8892 - 52ms/epoch - 4ms/step
Epoch 92/150
12/12 - 0s - loss: 0.0409 - accuracy: 0.9905 - val_loss: 0.7387 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 93/150
12/12 - 0s - loss: 0.0368 - accuracy: 0.9899 - val_loss: 0.6330 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 94/150
12/12 - 0s - loss: 0.0316 - accuracy: 0.9905 - val_loss: 0.6373 - val_accuracy: 0.8973 - 49ms/epoch - 4ms/step
Epoch 95/150
12/12 - 0s - loss: 0.0370 - accuracy: 0.9905 - val_loss: 0.6928 - val_accuracy: 0.9027 - 50ms/epoch - 4ms/step
Epoch 96/150
12/12 - 0s - loss: 0.0308 - accuracy: 0.9932 - val_loss: 0.7115 - val_accuracy: 0.8892 - 51ms/epoch - 4ms/step
Epoch 97/150
12/12 - 0s - loss: 0.0257 - accuracy: 0.9946 - val_loss: 0.6610 - val_accuracy: 0.8892 - 49ms/epoch - 4ms/step
Epoch 98/150
12/12 - 0s - loss: 0.0250 - accuracy: 0.9946 - val_loss: 0.7108 - val_accuracy: 0.8838 - 49ms/epoch - 4ms/step
Epoch 99/150
12/12 - 0s - loss: 0.0176 - accuracy: 0.9953 - val_loss: 0.8109 - val_accuracy: 0.8838 - 50ms/epoch - 4ms/step
Epoch 100/150
12/12 - 0s - loss: 0.0391 - accuracy: 0.9919 - val_loss: 0.7554 - val_accuracy: 0.8865 - 50ms/epoch - 4ms/step
Epoch 101/150
12/12 - 0s - loss: 0.0382 - accuracy: 0.9892 - val_loss: 0.7430 - val_accuracy: 0.8892 - 50ms/epoch - 4ms/step
Epoch 102/150
12/12 - 0s - loss: 0.0268 - accuracy: 0.9899 - val_loss: 0.7760 - val_accuracy: 0.8946 - 51ms/epoch - 4ms/step
Epoch 103/150
12/12 - 0s - loss: 0.0156 - accuracy: 0.9973 - val_loss: 0.7459 - val_accuracy: 0.8892 - 50ms/epoch - 4ms/step
Epoch 104/150
12/12 - 0s - loss: 0.0150 - accuracy: 0.9959 - val_loss: 0.7902 - val_accuracy: 0.8865 - 49ms/epoch - 4ms/step
Epoch 105/150
12/12 - 0s - loss: 0.0114 - accuracy: 0.9986 - val_loss: 0.7148 - val_accuracy: 0.8919 - 49ms/epoch - 4ms/step
Epoch 106/150
12/12 - 0s - loss: 0.0175 - accuracy: 0.9959 - val_loss: 0.7149 - val_accuracy: 0.8865 - 49ms/epoch - 4ms/step
Epoch 107/150
12/12 - 0s - loss: 0.0137 - accuracy: 0.9966 - val_loss: 0.7227 - val_accuracy: 0.9027 - 50ms/epoch - 4ms/step
Epoch 108/150
12/12 - 0s - loss: 0.0088 - accuracy: 0.9986 - val_loss: 0.6563 - val_accuracy: 0.9081 - 50ms/epoch - 4ms/step
Epoch 109/150
12/12 - 0s - loss: 0.0225 - accuracy: 0.9973 - val_loss: 0.7039 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 110/150
12/12 - 0s - loss: 0.0168 - accuracy: 0.9966 - val_loss: 0.6967 - val_accuracy: 0.9000 - 49ms/epoch - 4ms/step
Epoch 111/150
12/12 - 0s - loss: 0.0180 - accuracy: 0.9946 - val_loss: 0.7362 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 112/150
12/12 - 0s - loss: 0.0183 - accuracy: 0.9959 - val_loss: 0.7327 - val_accuracy: 0.9000 - 50ms/epoch - 4ms/step
Epoch 113/150
12/12 - 0s - loss: 0.0188 - accuracy: 0.9926 - val_loss: 0.7191 - val_accuracy: 0.8973 - 50ms/epoch - 4ms/step
Epoch 114/150
12/12 - 0s - loss: 0.0111 - accuracy: 0.9993 - val_loss: 0.7283 - val_accuracy: 0.8946 - 51ms/epoch - 4ms/step
Epoch 115/150
12/12 - 0s - loss: 0.0137 - accuracy: 0.9959 - val_loss: 0.7508 - val_accuracy: 0.8838 - 50ms/epoch - 4ms/step
Epoch 116/150
12/12 - 0s - loss: 0.0166 - accuracy: 0.9993 - val_loss: 0.6597 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 117/150
12/12 - 0s - loss: 0.0157 - accuracy: 0.9953 - val_loss: 0.7063 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 118/150
12/12 - 0s - loss: 0.0090 - accuracy: 0.9993 - val_loss: 0.7435 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 119/150
12/12 - 0s - loss: 0.0173 - accuracy: 0.9966 - val_loss: 0.8630 - val_accuracy: 0.8946 - 49ms/epoch - 4ms/step
Epoch 120/150
12/12 - 0s - loss: 0.0217 - accuracy: 0.9953 - val_loss: 0.6742 - val_accuracy: 0.8892 - 49ms/epoch - 4ms/step
Epoch 121/150
12/12 - 0s - loss: 0.0136 - accuracy: 0.9973 - val_loss: 0.6311 - val_accuracy: 0.9000 - 50ms/epoch - 4ms/step
Epoch 122/150
12/12 - 0s - loss: 0.0139 - accuracy: 0.9966 - val_loss: 0.7327 - val_accuracy: 0.8973 - 49ms/epoch - 4ms/step
Epoch 123/150
12/12 - 0s - loss: 0.0082 - accuracy: 0.9993 - val_loss: 0.8027 - val_accuracy: 0.9054 - 50ms/epoch - 4ms/step
Epoch 124/150
12/12 - 0s - loss: 0.0216 - accuracy: 0.9953 - val_loss: 0.6718 - val_accuracy: 0.9108 - 51ms/epoch - 4ms/step
Epoch 125/150
12/12 - 0s - loss: 0.0190 - accuracy: 0.9946 - val_loss: 0.7560 - val_accuracy: 0.9000 - 50ms/epoch - 4ms/step
Epoch 126/150
12/12 - 0s - loss: 0.0155 - accuracy: 0.9966 - val_loss: 0.6962 - val_accuracy: 0.9000 - 50ms/epoch - 4ms/step
Epoch 127/150
12/12 - 0s - loss: 0.0130 - accuracy: 0.9966 - val_loss: 0.7704 - val_accuracy: 0.8973 - 51ms/epoch - 4ms/step
Epoch 128/150
12/12 - 0s - loss: 0.0194 - accuracy: 0.9946 - val_loss: 0.7877 - val_accuracy: 0.8919 - 49ms/epoch - 4ms/step
Epoch 129/150
12/12 - 0s - loss: 0.0260 - accuracy: 0.9973 - val_loss: 0.7946 - val_accuracy: 0.9027 - 50ms/epoch - 4ms/step
Epoch 130/150
12/12 - 0s - loss: 0.0216 - accuracy: 0.9953 - val_loss: 0.9001 - val_accuracy: 0.8865 - 50ms/epoch - 4ms/step
Epoch 131/150
12/12 - 0s - loss: 0.0100 - accuracy: 0.9980 - val_loss: 0.7437 - val_accuracy: 0.9000 - 54ms/epoch - 5ms/step
Epoch 132/150
12/12 - 0s - loss: 0.0273 - accuracy: 0.9912 - val_loss: 0.8239 - val_accuracy: 0.8892 - 50ms/epoch - 4ms/step
Epoch 133/150
12/12 - 0s - loss: 0.0183 - accuracy: 0.9953 - val_loss: 0.7347 - val_accuracy: 0.8946 - 51ms/epoch - 4ms/step
Epoch 134/150
12/12 - 0s - loss: 0.0195 - accuracy: 0.9973 - val_loss: 0.7622 - val_accuracy: 0.8919 - 50ms/epoch - 4ms/step
Epoch 135/150
12/12 - 0s - loss: 0.0298 - accuracy: 0.9912 - val_loss: 0.7912 - val_accuracy: 0.8811 - 49ms/epoch - 4ms/step
Epoch 136/150
12/12 - 0s - loss: 0.0346 - accuracy: 0.9885 - val_loss: 0.7766 - val_accuracy: 0.8838 - 50ms/epoch - 4ms/step
Epoch 137/150
12/12 - 0s - loss: 0.0341 - accuracy: 0.9939 - val_loss: 0.7249 - val_accuracy: 0.8946 - 50ms/epoch - 4ms/step
Epoch 138/150
12/12 - 0s - loss: 0.0218 - accuracy: 0.9959 - val_loss: 0.6659 - val_accuracy: 0.9000 - 50ms/epoch - 4ms/step
Epoch 139/150
12/12 - 0s - loss: 0.0147 - accuracy: 0.9966 - val_loss: 0.6719 - val_accuracy: 0.8892 - 49ms/epoch - 4ms/step
Epoch 140/150
12/12 - 0s - loss: 0.0202 - accuracy: 0.9946 - val_loss: 0.6815 - val_accuracy: 0.9000 - 51ms/epoch - 4ms/step
Epoch 141/150
12/12 - 0s - loss: 0.0219 - accuracy: 0.9946 - val_loss: 0.7254 - val_accuracy: 0.8838 - 51ms/epoch - 4ms/step
Epoch 142/150
12/12 - 0s - loss: 0.0254 - accuracy: 0.9905 - val_loss: 0.6758 - val_accuracy: 0.8892 - 48ms/epoch - 4ms/step
Epoch 143/150
12/12 - 0s - loss: 0.0200 - accuracy: 0.9953 - val_loss: 0.5930 - val_accuracy: 0.9081 - 51ms/epoch - 4ms/step
Epoch 144/150
12/12 - 0s - loss: 0.0157 - accuracy: 0.9946 - val_loss: 0.6040 - val_accuracy: 0.9081 - 51ms/epoch - 4ms/step
Epoch 145/150
12/12 - 0s - loss: 0.0132 - accuracy: 0.9986 - val_loss: 0.7273 - val_accuracy: 0.8973 - 50ms/epoch - 4ms/step
Epoch 146/150
12/12 - 0s - loss: 0.0134 - accuracy: 0.9980 - val_loss: 0.6897 - val_accuracy: 0.9000 - 53ms/epoch - 4ms/step
Epoch 147/150
12/12 - 0s - loss: 0.0324 - accuracy: 0.9899 - val_loss: 0.6122 - val_accuracy: 0.8892 - 50ms/epoch - 4ms/step
Epoch 148/150
12/12 - 0s - loss: 0.0209 - accuracy: 0.9939 - val_loss: 0.7285 - val_accuracy: 0.9081 - 51ms/epoch - 4ms/step
Epoch 149/150
12/12 - 0s - loss: 0.0226 - accuracy: 0.9932 - val_loss: 0.6426 - val_accuracy: 0.9054 - 51ms/epoch - 4ms/step
Epoch 150/150
12/12 - 0s - loss: 0.0162 - accuracy: 0.9966 - val_loss: 0.7837 - val_accuracy: 0.8946 - 52ms/epoch - 4ms/step
"""
# 1. Extração dos Dados (sem alterações)
history_loss = []
history_accuracy = []
history_val_loss = []
history_val_accuracy = []

pattern = re.compile(r"loss: ([\d.]+) - accuracy: ([\d.]+) - val_loss: ([\d.]+) - val_accuracy: ([\d.]+)")

for line in training_log.strip().split('\n'):
    match = pattern.search(line)
    if match:
        history_loss.append(float(match.group(1)))
        history_accuracy.append(float(match.group(2)))
        history_val_loss.append(float(match.group(3)))
        history_val_accuracy.append(float(match.group(4)))

epochs = range(1, len(history_loss) + 1)
plt.style.use('seaborn-v0_8-whitegrid')
best_epoch_loss = np.argmin(history_val_loss) + 1


# --- ALTERAÇÃO AQUI: Plotagem do Gráfico de Perda em um arquivo separado ---

# Cria a primeira figura, apenas para o gráfico de Perda
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(epochs, history_loss, 'o-', color='royalblue', label='Perda de Treinamento', markersize=4, alpha=0.7)
ax1.plot(epochs, history_val_loss, 's-', color='darkorange', label='Perda de Validação', markersize=4, alpha=0.7)
ax1.set_xlabel('Épocas', fontsize=12)
ax1.set_ylabel('Perda (Loss)', fontsize=12)
#ax1.set_title('Histórico de Perda Durante o Treinamento', fontsize=16)
ax1.axvline(x=best_epoch_loss, color='crimson', linestyle='--', label=f'Melhor Época ({best_epoch_loss})')
ax1.legend()

# Salva a primeira figura
plt.tight_layout()
plt.savefig("grafico_perda.png", dpi=300)
print(f"Gráfico de Perda salvo com sucesso no arquivo 'grafico_perda.png'")
# plt.close(fig1) # Fecha a figura para liberar memória


# --- ALTERAÇÃO AQUI: Plotagem do Gráfico de Acurácia em um arquivo separado ---

# Cria a segunda figura, apenas para o gráfico de Acurácia
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.plot(epochs, history_accuracy, 'o-', color='royalblue', label='Acurácia de Treinamento', markersize=4, alpha=0.7)
ax2.plot(epochs, history_val_accuracy, 's-', color='darkorange', label='Acurácia de Validação', markersize=4, alpha=0.7)
ax2.set_xlabel('Épocas', fontsize=12)
ax2.set_ylabel('Acurácia', fontsize=12)
#ax2.set_title('Histórico de Acurácia Durante o Treinamento', fontsize=16)
ax2.legend()
# Melhora a formatação do eixo Y para porcentagem
ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

# Salva a segunda figura
plt.tight_layout()
plt.savefig("grafico_acuracia.png", dpi=300)
print(f"Gráfico de Acurácia salvo com sucesso no arquivo 'grafico_acuracia.png'")

# Exibe os gráficos (opcional, pode ser removido se a execução for apenas para salvar)
plt.show()