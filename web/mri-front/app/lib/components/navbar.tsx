'use client';
import { FunctionComponent, useCallback } from "react";
import { useNavigate } from "react-router-dom";

export type NavbarType = {
  className?: string;

  /** Variant props */
  breakpoint?: string;
};

const Navbar: FunctionComponent<NavbarType> = ({
  className = "",
  breakpoint = "Desktop",
}) => {
  const navigate = useNavigate();

  const onLinkTextClick = useCallback(() => {
    navigate("/about");
  }, [navigate]);

  return (
    <div
      className={`flex-1 bg-background-color-primary border-color-neutral-black border-b-[1px] border-solid box-border flex flex-col items-center justify-center py-0 px-16 min-h-[72px] max-w-full text-left text-base text-color-neutral-black font-heading-desktop-h1 mq725:pl-8 mq725:pr-8 mq725:box-border ${className}`}
      data-breakpoint={breakpoint}
    >
      <div className="self-stretch flex flex-row items-center justify-between max-w-full gap-5 mq650:flex-wrap">
        <div className="h-10 flex flex-row items-center justify-center">
          <img
            className="h-9 w-[84px] relative overflow-hidden shrink-0"
            loading="lazy"
            alt=""
            src="/company-logo.svg"
          />
        </div>
        <div className="flex flex-row items-center justify-center gap-8 max-w-full mq450:gap-4 mq450:flex-wrap">
          <div className="flex flex-row items-center justify-between gap-5">
            <div className="flex flex-row items-center justify-center">
              <button
                className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-base leading-[150%] font-heading-desktop-h1 text-color-neutral-black text-left inline-block"
                onClick={onLinkTextClick}
              >
                Link One
              </button>
            </div>
            <div className="flex flex-row items-center justify-center">
              <button
                className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-base leading-[150%] font-heading-desktop-h1 text-color-neutral-black text-left inline-block"
                onClick={onLinkTextClick}
              >
                Link Two
              </button>
            </div>
            <div className="self-stretch hidden flex-row items-center justify-center">
              <div className="self-stretch relative leading-[150%]">
                Link Three
              </div>
            </div>
            <div className="self-stretch hidden flex-row items-center justify-center relative gap-1">
              <div className="self-stretch flex flex-row items-center justify-start gap-1 z-[0]">
                <div className="self-stretch relative leading-[150%]">
                  Link Four
                </div>
                <img
                  className="h-6 w-6 relative overflow-hidden shrink-0 min-h-[24px]"
                  alt=""
                  src="/chevron-down.svg"
                />
              </div>
              <div className="!m-[0] absolute top-[32px] left-[0px] bg-background-color-primary border-color-neutral-black border-[1px] border-solid overflow-hidden hidden flex-col items-start justify-start py-[22px] pl-6 pr-5 gap-4 z-[1]">
                <div className="self-stretch h-6 relative leading-[150%] inline-block">
                  Link Five
                </div>
                <div className="self-stretch h-6 relative leading-[150%] inline-block">
                  Link Six
                </div>
                <div className="self-stretch h-6 relative leading-[150%] inline-block">
                  Link Seven
                </div>
              </div>
            </div>
          </div>
          <div className="h-10 flex flex-row items-center justify-center gap-4">
            <button
              className="cursor-pointer border-color-neutral-black border-[1px] border-solid py-2 px-5 bg-[transparent] h-[42px] box-border flex flex-row items-center justify-center hover:bg-darkslategray-200 hover:border-darkslategray-100 hover:border-[1px] hover:border-solid hover:box-border"
              onClick={onLinkTextClick}
            >
              <div className="relative text-base leading-[150%] font-heading-desktop-h1 text-color-neutral-black text-left">
                Button
              </div>
            </button>
            <button
              className="cursor-pointer border-color-neutral-black border-[1px] border-solid py-2 px-5 bg-color-neutral-black h-[42px] box-border flex flex-row items-center justify-center hover:bg-darkslategray-100 hover:border-darkslategray-100 hover:border-[1px] hover:border-solid hover:box-border"
              onClick={onLinkTextClick}
            >
              <div className="relative text-base leading-[150%] font-heading-desktop-h1 text-background-color-primary text-left">
                Button
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
