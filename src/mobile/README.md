# indigenous_mobile_assistant

## Prerequisites

1. **Node.js**: Install Node.js (v20.x is recommended).  
   [Download and install Node.js](https://nodejs.org/).

2. **Java Development Kit (JDK)**: Install JDK 17.  
   [Download and install JDK](https://adoptopenjdk.net/).

3. **React Native CLI**: Make sure the React Native CLI is installed.

   ```bash
   npm install -g react-native-cli
   ```

4. **Gradlew Permissions**: Ensure ./gradlew has execute permissions:

   ```bash
   chmod +x ./gradlew
   ```

5. **Android Studio (for Local Testing)**: To run the app locally on an Android device or emulator, follow the steps described in [Android development environment](https://reactnative.dev/docs/set-up-your-environment) to install and configure Android Studio.

## Change the version of the application

The application version is controlled by a text file called version.txt located in the android/app/src/main/assets/version.txt folder.

## Deploy the app to Google Console

1. Clone the repository:

   ```bash
    git clone https://github.ibm.com/BRL-indigenous/indigenous_mobile_assistant.git
    cd indigenous_mobile_assistant
   ```

2. Install dependencies:

   ```bash
   npm install --legacy-peer-deps
   ```

3. Navigate to the Android folder:

   ```bash
   cd android
   ```

4. Change the versionCode in android/app/build.gradle

   Before building the APK we need to change the versionCode variable to the version later than the current one
   For example:
      versionCode 48 -> versionCode 49

4. Clean and assemble the APK.

   ```bash
   ./gradlew bundleRelease
   ```

6. The release bundle will be generated in:
   ```bash
   android/app/build/outputs/bundle/release/app-release.aab
   ```

7. Deploy

   Access your GooglePlay developer account and create a new "Teste Interno" version

## Testing the Application Locally

1. **Install Dependencies**

Before running the application locally, ensure that you have installed the prerequisites as described in the project documentation.

2. **Clone the repository**

```bash
git clone https://github.ibm.com/BRL-indigenous/indigenous_mobile_assistant.git
npm install --legacy-peer-deps
```

3. **Start the development server**

```bash
npm start
```


