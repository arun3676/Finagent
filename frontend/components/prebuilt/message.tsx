import Markdown from "react-markdown";
import { cn } from "../../utils/utils";

export interface MessageTextProps {
  content: string;
}

export function AIMessageText(props: MessageTextProps) {
  return (
    <div className="flex mr-auto w-fit max-w-[80%] md:max-w-[700px] bg-secondary text-secondary-foreground rounded-lg rounded-tl-none px-4 py-2 mt-2">
      <div className="text-sm prose dark:prose-invert break-words">
        <Markdown>{props.content}</Markdown>
      </div>
    </div>
  );
}

export function HumanMessageText(props: MessageTextProps) {
  return (
    <div className="flex ml-auto w-fit max-w-[80%] md:max-w-[700px] bg-primary text-primary-foreground rounded-lg rounded-tr-none px-4 py-2">
      <p className="text-sm break-words">
        {props.content}
      </p>
    </div>
  );
}
