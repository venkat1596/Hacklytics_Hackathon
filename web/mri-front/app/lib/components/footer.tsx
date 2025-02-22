import { FunctionComponent } from "react";

export type FooterType = {
  className?: string;

  /** Variant props */
  breakpoint?: string;
};

const Footer: FunctionComponent<FooterType> = ({
  className = "",
  breakpoint = "Desktop",
}) => {
  return (
    <section
      className={`flex-1 bg-background-color-primary overflow-hidden flex flex-col items-center justify-start py-20 px-16 box-border gap-20 max-w-full text-left text-sm text-color-neutral-black font-heading-desktop-h1 mq450:gap-5 mq725:gap-10 mq725:pl-8 mq725:pr-8 mq725:box-border ${className}`}
      data-breakpoint={breakpoint}
    >
      <div className="self-stretch flex flex-row items-center justify-start flex-wrap content-center gap-8 max-w-full mq650:gap-4">
        <div className="flex-1 overflow-hidden flex flex-col items-start justify-start min-w-[310px] max-w-full">
          <img
            className="w-[84px] h-9 relative overflow-hidden shrink-0"
            loading="lazy"
            alt=""
            src="/company-logo.svg"
          />
        </div>
        <div className="flex-1 flex flex-row items-start justify-between min-w-[293px] max-w-full gap-5 mq450:flex-wrap">
          <button className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-sm leading-[150%] font-semibold font-heading-desktop-h1 text-color-neutral-black text-left inline-block">
            Link One
          </button>
          <button className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-sm leading-[150%] font-semibold font-heading-desktop-h1 text-color-neutral-black text-left inline-block">
            Link Two
          </button>
          <button className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-sm leading-[150%] font-semibold font-heading-desktop-h1 text-color-neutral-black text-left inline-block">
            Link Three
          </button>
          <button className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-sm leading-[150%] font-semibold font-heading-desktop-h1 text-color-neutral-black text-left inline-block">
            Link Four
          </button>
          <button className="cursor-pointer [border:none] p-0 bg-[transparent] relative text-sm leading-[150%] font-semibold font-heading-desktop-h1 text-color-neutral-black text-left inline-block">
            Link Five
          </button>
        </div>
        <div className="flex-1 flex flex-row items-center justify-end gap-3 min-w-[120px] max-w-full mq450:flex-wrap mq450:justify-center">
          <img
            className="h-6 w-6 relative overflow-hidden shrink-0 min-h-[24px]"
            loading="lazy"
            alt=""
            src="/icon--facebook.svg"
          />
          <img
            className="h-6 w-6 relative overflow-hidden shrink-0 min-h-[24px]"
            loading="lazy"
            alt=""
            src="/icon--instagram.svg"
          />
          <img
            className="h-6 w-6 relative overflow-hidden shrink-0 min-h-[24px]"
            loading="lazy"
            alt=""
            src="/icon--x.svg"
          />
          <img
            className="h-6 w-6 relative overflow-hidden shrink-0 min-h-[24px]"
            loading="lazy"
            alt=""
            src="/icon--linkedin.svg"
          />
          <img
            className="h-6 w-6 relative overflow-hidden shrink-0 min-h-[24px]"
            loading="lazy"
            alt=""
            src="/icon--youtube.svg"
          />
        </div>
      </div>
      <div className="self-stretch flex flex-col items-center justify-start gap-8 max-w-full mq650:gap-4">
        <div className="self-stretch h-px relative bg-color-neutral-black border-color-neutral-black border-[1px] border-solid box-border" />
        <div className="flex flex-row items-start justify-start gap-6 max-w-full mq650:flex-wrap">
          <div className="relative leading-[150%]">
            © 2024 Relume. All rights reserved.
          </div>
          <div className="flex flex-row items-start justify-start gap-6 max-w-full mq450:flex-wrap">
            <div className="relative [text-decoration:underline] leading-[150%]">
              Privacy Policy
            </div>
            <div className="relative [text-decoration:underline] leading-[150%]">
              Terms of Service
            </div>
            <div className="relative [text-decoration:underline] leading-[150%]">
              Cookies Settings
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Footer;
