@echo off
echo ===================================
echo Building F.R.E.D Desktop Application
echo ===================================

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Install packaging requirements
echo Installing packaging dependencies...
pip install -r packaging_requirements.txt

:: Run PyInstaller
echo Building application with PyInstaller...
pyinstaller --clean fred.spec

:: Create a folder for the installer files
echo Creating installer directory...
if not exist "FRED_Installer" mkdir "FRED_Installer"

:: Copy the dist folder to the installer directory
echo Copying files to installer directory...
xcopy /E /I /Y "dist\FRED" "FRED_Installer\FRED"

:: Copy the post-installation script
echo Copying post-installation script...
copy "post_install.py" "FRED_Installer\FRED\"

:: Copy README
echo Copying documentation...
copy "README.md" "FRED_Installer\README.md"

:: Create a simple installer batch file
echo @echo off > "FRED_Installer\Install_FRED.bat"
echo cd FRED >> "FRED_Installer\Install_FRED.bat"
echo python post_install.py >> "FRED_Installer\Install_FRED.bat"
echo pause >> "FRED_Installer\Install_FRED.bat"

echo ===================================
echo Build Complete!
echo The installer can be found in the FRED_Installer directory.
echo ===================================

pause 