import PIL.Image


crgba_image = PIL.Image.open("styles_images/chiikawa.png")
grgba_image = PIL.Image.open("styles_images/gogon.png")
crgb_image = crgba_image.convert('RGB')
grgb_image = grgba_image.convert('RGB')

crgb_image.save("styles_images/chiikawa_rgb.png")
grgb_image.save("styles_images/gogon_rgb.png")
