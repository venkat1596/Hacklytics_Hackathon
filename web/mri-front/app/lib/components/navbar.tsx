import { FunctionComponent } from "react";
import Button from "./button";

export type Navbar1Type = {
  className?: string;

  /** Variant props */
  breakpoint?: string;
};

const Navbar1: FunctionComponent<Navbar1Type> = ({
  className = "",
  breakpoint = "Desktop",
}) => {
  return (
    <div
      className={`w-[1440px] bg-color-neutral-white border-color-neutral-black border-solid border-b-[1px] box-border h-[72px] overflow-hidden shrink-0 flex flex-col items-center justify-center py-0 px-16 text-left text-base text-color-neutral-black font-text-regular-normal ${className}`}
      data-breakpoint={breakpoint}
    >
      <div className="self-stretch flex flex-row items-center justify-center gap-8">
        <div className="flex-1 flex flex-row items-start justify-start">
          <img
            className="h-[27px] w-[63px] relative overflow-hidden shrink-0"
            loading="lazy"
            alt=""
            src="/company-logo.svg"
          />
        </div>
        <div className="overflow-hidden flex flex-row items-start justify-start gap-8">
          <div className="relative leading-[150%]">Demo</div>
          <div className="relative leading-[150%]">Our Process</div>
          <div className="relative leading-[150%]">About Us</div>
          <div className="flex flex-row items-center justify-center gap-1">
            <div className="relative leading-[150%]">Link Four</div>
            <img
              className="w-6 relative h-6 overflow-hidden shrink-0"
              alt=""
              src="/chevron-down.svg"
            />
          </div>
        </div>
        <div className="flex-1 flex flex-col items-end justify-center">
          <Button
            iconPosition="No icon"
            small
            style="Primary"
            button="Button"
          />
        </div>
      </div>
    </div>
  );
};

export default Navbar1;
