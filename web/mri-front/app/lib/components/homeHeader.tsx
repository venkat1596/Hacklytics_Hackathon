'use client'
import { FunctionComponent } from "react";
import Button from "./button";

export type HeaderType = {
  className?: string;

  /** Variant props */
  breakpoint?: string;
};

const Header: FunctionComponent<HeaderType> = ({
  className = "",
  breakpoint = "Desktop",
}) => {
  return (
    <section
      className={`w-[1440px] bg-color-neutral-white overflow-hidden shrink-0 flex flex-col items-center justify-start py-20 px-16 box-border ${className}`}
      data-breakpoint={breakpoint}
    >
      <header className="self-stretch flex flex-col items-start justify-start text-left text-37xl text-color-neutral-black font-text-regular-normal">
        <div className="w-[1312px] border-color-neutral-black border-solid border-[1px] box-border overflow-hidden flex flex-row items-center justify-start">
          <div className="self-stretch flex-[0.8537] flex flex-col items-start justify-center p-29xl gap-6">
            <div className="self-stretch flex flex-col items-start justify-start gap-6 pl-4"> 
              <h1 className="m-0 self-stretch relative text-inherit leading-[120%] font-bold font-[inherit]">
              Welcome to the MRI Cycle Gan Superimposer
              </h1>
              <div className="self-stretch relative text-lg leading-[150%]">
              Our tool enables medical profesionals to have access to cost prohibitive quality with relatively inexpenseive hardware. Check it out for yourself!
              </div>
            </div>
            <div className="flex flex-row items-start justify-start pt-4 px-0 pb-0 gap-4">
              <Button
                darkMode={false}
                iconPosition="No icon"
                small={false}
                style="Primary"
                button="Upload"
              />
              <Button
                darkMode={false}
                iconPosition="No icon"
                small={false}
                style="Secondary"
                button="Run?"   //Loe key their may not be a purpose for this button. 
              />
            </div>
          </div>
          <div className="flex-1 h-[640px] flex flex-col items-center justify-start">
            <img
              className="self-stretch flex-1 relative max-w-full overflow-hidden max-h-full object-cover"
              loading="lazy"
              alt=""
              src="/placeholder-image1@2x.png"
            />
          </div>
        </div>
      </header>
    </section>
  );
};

export default Header;
