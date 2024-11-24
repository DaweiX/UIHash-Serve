## UIHash

### TL;TR

Just simply run the end-to-end flow by
```
python main.py
```

</br>

***

***

### 1. APK to Raw UI Data

Run qemu Android emulator:

```sh
bash collect/run_qemu.sh
```

In a new terminal, run app dynamic analysis:

```sh
conda activate platform
python collect/main.py xxx.apk
```

Where `xxx.apk` is an apk file in `config.path.APK`.
This will output a folder named by the sha256 value of the apk under the path `config.path.config.path.data`.
In this folder, there are several xml-jpg pairs, each pair corresponds to the runtime hierarchy and screenshot image of a UI.

### 2. Raw UI Data to UI#

Simply run `ui2hash/main.py`.
The input is an folder that contains UI information, and you can see the output in the console, which is a list
Each item in the list is a tuple with four elements:  

1. SHA256 value of the apk  
2. Name of the UI's activity  
3. Reidentified view types  
4. UI# array, i.e., the UI representation  

### 3. Get Similarity Results
In `uihash/compare.py`, the function `search_similar_uis` takes an apk SHA256 as input, and output the similarity detection results for each of its UIs.

Here is a sample detection result output printed by the function `print_results`:

```
----------------------------
Top 5 similar UIs detected
----------------------------
UI: com.droidgamers.news-com.appyet.mobile.activity.MainActivity (APK: 1DE2FA20BB138CFE9460689C5458A672AA8C348D3DE362FA4C933F00B0469917
  Similar UI: com.actualidadgoogle.news-com.appyet.mobile.activity.MainActivity (APK: 29E07BBABC262CBA61B5A14E8DCA18FA6B21C1E5BA31D28AF17B922702A0F157, Score: 0.8715)
  Similar UI: com.softek.ofxclmobile.wyhyfcu-com.softek.ofxclmobile.FormAgreement (APK: AAFEF9C38659E5EE562C853A5A30AE0CECF005DFCE3C822440FA46B787D1F50A, Score: 0.7339)
  Similar UI: com.softek.ofxclmobile.wyhyfcu-com.softek.ofxclmobile.FormSplash (APK: AAFEF9C38659E5EE562C853A5A30AE0CECF005DFCE3C822440FA46B787D1F50A, Score: 0.7339)
  Similar UI: com.idrettirana-com.appyet.mobile.activity.MainActivity (APK: D56022EB2C2E02E47A4FF83433E247122169E5E93290992CC34723A2E1C1CA92, Score: 0.6892)
  Similar UI: com.softek.ofxclmobile.wsecu-com.softek.ofxclmobile.FormAgreement (APK: 2AD2E21F08BF07C9B734FFF53DCE7BD8FD62AF26F8D1888FE61221020A7C14AC, Score: 0.5467)
UI: com.droidgamers.news-com.appyet.mobile.activity.SearchActivity (APK: 1DE2FA20BB138CFE9460689C5458A672AA8C348D3DE362FA4C933F00B0469917
  Similar UI: com.idrettirana-com.appyet.mobile.activity.SearchActivity (APK: D56022EB2C2E02E47A4FF83433E247122169E5E93290992CC34723A2E1C1CA92, Score: 1.0000)
  Similar UI: com.yahooTW.news.reader.anubis-com.appyet.mobile.activity.SearchActivity (APK: A277B0D16837D1F1D7B563DDC2A4243379D9F57E4410BB72B49B16422069AB3F, Score: 1.0000)
  Similar UI: com.actualidadgoogle.news-com.appyet.mobile.activity.SearchActivity (APK: 29E07BBABC262CBA61B5A14E8DCA18FA6B21C1E5BA31D28AF17B922702A0F157, Score: 0.9339)
  Similar UI: com.idrettirana-com.appyet.mobile.activity.MainActivity (APK: D56022EB2C2E02E47A4FF83433E247122169E5E93290992CC34723A2E1C1CA92, Score: 0.7834)
  Similar UI: com.burp.sounds-com.qbiki.location.LocationDetectorActivity (APK: 16A806B12FA54C02EFFEB854F64775B496196A3B3D23486A5B7E27B15A10AFBB, Score: 0.6803)
```

</br>

***

***

### Fresh Setup Steps

1. Config Python Packages. To make it easier, run

    ```
    conda env create -f environment.yml
    ```

    This will install some python packages including
    + pyyaml
    + torch, torchvision
    + colorlog
    + faiss-cpu
    + ...



2. Install MySQL
```bash
sudo apt install mysql-server
sudo mysql
```

After entering mysql, create a new mysql account, and grant permissions:

```bash
CREATE DATABASE <database_name>;
CREATE USER '<user>'@'localhost' IDENTIFIED BY '<new_password>';
GRANT ALL PRIVILEGES ON <database_name>.* TO '<user>'@'localhost';
```

Make sure the placeholder values are the same as those in `config.yml`.
Then, run 

```bash
python main.py INITDB
```

To init the database structure.

3. Install QEMU