import React from 'react';
import { SvgXml } from 'react-native-svg';

interface SVGIconProps {
  xml: string;
  width?: number;
  height?: number;
  color?: string;
}

const SVGIcon: React.FC<SVGIconProps> = ({ xml, width = 24, height = 24, color = 'currentColor' }) => {
  const updatedXml = xml.replace(/currentColor/g, color);

  return <SvgXml xml={updatedXml} width={width} height={height} />;
};

export default SVGIcon;