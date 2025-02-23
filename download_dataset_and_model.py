import gdown
import shutil

gdown.download(id="1tSqCqj4th5FU_luLU0ogr--HaQFBpeIV")
gdown.download(id="1BkhIwjo94DLTmkI93-QgydJiq7G_7byS")
gdown.download(id="19GfihNmKJC9ndAv3EVxG2YuOHNVrJ8a2")
print("Unpacking dataset.zip...")
shutil.unpack_archive("dataset.zip", "./nrp/")
print("Unpacking models.zip...")
shutil.unpack_archive("models.zip", "./nrp/")
print("Unpacking fire_dataset.zip...")
shutil.unpack_archive("fire_dataset.zip", "./nrp/")
