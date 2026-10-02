# PixelForge_4x_RRDB-S

- **Task:** 4x super-resolution of game textures (albedo / colour maps)
- **Architecture:** ESRGAN RRDBNet, 6 RRDB blocks, 32 features, grow 16 (1.12M params)
- **Training data:** Poly Haven textures (CC0), realistic degradations (blur, downscale, noise, JPEG); L1 then GAN stage
- **Licence:** PolyForm Noncommercial 1.0.0 (free for modding and personal use; commercial licence on request)
- **Source:** https://github.com/Priyanshu-Singh-git/pixelforge
- **Not for:** pixel-art sprites, normal maps, text-heavy UI textures
