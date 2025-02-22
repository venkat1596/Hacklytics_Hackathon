import React from 'react';
import { HeaderSimple } from '../lib/HeaderSimple';
import { FooterCentered } from '../lib/FooterCentered';
import { FeaturesImages } from '../lib/FeaturesImages';
import { HeroText } from '../lib/HeroText';
import { NavbarSimpleColored } from '../lib/SideBar'; // Import the navbar
import styles from '../page.module.css'; // Import the same styles

const AboutPage: React.FC = () => {
  return (
    <div className={styles.container}>
      <NavbarSimpleColored /> {/* Add the navbar */}
      <div className={styles.content}>
        <HeaderSimple />
        <main>
          <HeroText />
          <FeaturesImages />
        </main>
      </div>
    </div>
  );
};

export default AboutPage;