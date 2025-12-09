import { useState } from "react";
import { Instruction } from "./instructions";
import { FaCircleArrowRight } from "react-icons/fa6";
import { FaCircleArrowLeft } from "react-icons/fa6";
import { MdKeyboardBackspace } from "react-icons/md";

type HelperProps = {
  instructions: Instruction[];
  setInstructions: React.Dispatch<React.SetStateAction<Instruction[]>>;
};

export default function Helper({ instructions, setInstructions }: HelperProps) {
  const [currentInstruction, setCurrentInstruction] = useState(0);

  return (
    <div className="h-full w-full flex items-center justify-center gap-4 max-w-[500px] relative bg-gray-100 pb-4">
      <MdKeyboardBackspace
        className="transition-all bg-gray-100 absolute top-0 left-0 text-2xl lg:text-3xl text-blue-500 hover:text-blue-700 cursor-pointer"
        onClick={() => setInstructions([])}
      />

      <div className="h-full bg-gray-100 w-full flex flex-col gap-4">
        <div className="flex items-center justify-center">
          <FaCircleArrowLeft
            className="transition-all text-3xl lg:text-4xl text-blue-500 hover:text-blue-700 cursor-pointer"
            onClick={() => {
              if (currentInstruction > 0)
                setCurrentInstruction(currentInstruction - 1);
            }}
          />
          <img
            src={instructions[currentInstruction].image}
            className="object-contain h-[400px] w-[300px] lg:h-[500px] lg:w-[400px]"
          />
          <FaCircleArrowRight
            onClick={() => {
              if (currentInstruction < instructions.length - 1)
                setCurrentInstruction(currentInstruction + 1);
            }}
            className="transition-all text-3xl lg:text-4xl text-blue-500 hover:text-blue-700 cursor-pointer"
          />
        </div>
        <p className="text-center bg-gray-100 ">
          {instructions[currentInstruction].description}
        </p>
      </div>
    </div>
  );
}
