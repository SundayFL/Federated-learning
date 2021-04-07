# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['server.py'],
             pathex=['/home/piotr/Desktop/MGR/FL-repo/Federated-learning/fl_image_clasification/pysyft/my/server'],
             binaries=[],
             datas=[('syft_proto', 'syft_proto'), ('shaloop', 'shaloop')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='server',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
