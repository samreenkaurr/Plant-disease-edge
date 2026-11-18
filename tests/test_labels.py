import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from common import map_folder_to_label as m

def test_plantvillage_names():
    assert m("Tomato___Early_blight") == "tomato_early_blight"
    assert m("Tomato___healthy") == "tomato_healthy"
    assert m("Potato___Late_blight") == "potato_late_blight"
    assert m("Tomato___Septoria_leaf_spot") == "tomato_septoria_leaf_spot"

def test_plantdoc_names():
    assert m("Tomato Early blight leaf") == "tomato_early_blight"
    assert m("Tomato leaf late blight") == "tomato_late_blight"
    assert m("Tomato leaf") == "tomato_healthy"
    assert m("Potato leaf early blight") == "potato_early_blight"
    assert m("Potato leaf") == "potato_healthy"

def test_unmapped():
    assert m("Tomato___Bacterial_spot") is None
    assert m("Tomato leaf bacterial spot") is None
    assert m("Apple___healthy") is None
    assert m("Potato leaf septoria") is None
