import { Dimensions, PixelRatio } from "react-native";

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get("window");
const BASE_WIDTH = 375;
const SCALE = SCREEN_WIDTH > SCREEN_HEIGHT ? SCREEN_HEIGHT : SCREEN_WIDTH;

const fontConfig = {
  phone: {
    small: { min: 0.8, max: 1 },
    medium: { min: 0.9, max: 1.1 },
    large: { min: 1, max: 1.2 },
  },
  tablet: {
    small: { min: 0.9, max: 1.0 },
    medium: { min: 1.0, max: 1.1 },
    large: { min: 1.6, max: 1.8 },
  },
};

const getDeviceType = (): "phone" | "tablet" => {
  const pixelDensity = PixelRatio.get();
  const adjustedWidth = SCREEN_WIDTH * pixelDensity;
  const adjustedHeight = SCREEN_HEIGHT * pixelDensity;
  if (pixelDensity < 2 && (adjustedWidth >= 1000 || adjustedHeight >= 1000)) {
    return "tablet";
  } else if (
    pixelDensity === 2 &&
    (adjustedWidth >= 1920 || adjustedHeight >= 1920)
  ) {
    return "tablet";
  } else {
    return "phone";
  }
};

const getScreenSizeCategory = (): "small" | "medium" | "large" => {
  if (SCALE < 350) return "small";
  if (SCALE > 500) return "large";
  return "medium";
};

export const responsiveFontSize = (size: number): number => {
  const deviceType = getDeviceType();
  const screenSizeCategory = getScreenSizeCategory();
  const config = fontConfig[deviceType][screenSizeCategory];
  const scaleFactor = SCALE / BASE_WIDTH;
  const clampedScaleFactor = Math.min(
    Math.max(scaleFactor, config.min),
    config.max
  );
  let newSize = size * clampedScaleFactor;
  if (deviceType === "tablet") {
    newSize *= 1.3;
  }
  return (
    Math.round(PixelRatio.roundToNearestPixel(newSize)) /
    PixelRatio.getFontScale()
  );
};
