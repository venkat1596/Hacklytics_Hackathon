// src/about/page.tsx
import React from 'react';
import { HeaderSimple } from '../lib/HeaderSimple';
import { FooterCentered } from '../lib/FooterCentered';
import { FeaturesImages } from '../lib/FeaturesImages';
import { HeroText } from '../lib/HeroText';
import { FeaturesCards } from '../lib/FeaturesCards';
const AboutPage: React.FC = () => {
  return (
    <div>
      <HeaderSimple />
      <main>
        <HeroText />
        <FeaturesImages />
        <FeaturesCards />
      </main>
      <FooterCentered />
    </div>
  );
};

export default AboutPage;
