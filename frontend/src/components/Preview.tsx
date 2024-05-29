import { useEffect, useRef } from "react";
import classNames from "classnames";
import useThrottle from "../hooks/useThrottle";

interface Props {
  code: string;
  device: "mobile" | "desktop";
}

function Preview({ code, device }: Props) {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);

  // Don't update code more often than every 200ms.
  const throttledCode = useThrottle(code, 200);

  // useEffect(() => {
  //   if (iframeRef.current) {
  //     iframeRef.current.srcdoc = throttledCode;
  //   }
  // }, [throttledCode]);

  useEffect(() => {
    const handleIframeLoad = () => {
      if (iframeRef.current) {
        iframeRef.current.contentWindow?.scrollTo({
          top: iframeRef.current.contentDocument?.body.scrollHeight,
          behavior: 'smooth'
        });
      }
    };

    if (iframeRef.current) {
      iframeRef.current.srcdoc = throttledCode;
      iframeRef.current.addEventListener('load', handleIframeLoad);
      return () => {
        iframeRef.current?.removeEventListener('load', handleIframeLoad);
      };
    }
  }, [throttledCode]);

  
  return (
    <div className="flex justify-center mx-2">
      <iframe
        id={`preview-${device}`}
        ref={iframeRef}
        title="Preview"
        className={classNames(
          "border-[4px] border-black rounded-[20px] shadow-lg",
          "transform scale-[0.9] origin-top",
          {
            "w-full h-[832px]": device === "desktop",
            "w-[400px] h-[832px]": device === "mobile",
          }
        )}
      ></iframe>
    </div>
  );
}

export default Preview;
