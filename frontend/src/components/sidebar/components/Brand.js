import React from "react";
import { Flex, useColorModeValue, Box } from "@chakra-ui/react";
import { BirdHouseIcon } from "components/icons/Icons";
import { HSeparator } from "components/separator/Separator";

export function SidebarBrand() {
  let logoColor = useColorModeValue("navy.700", "white");

  return (
    <Flex align="center" direction="column" position="relative" overflow="hidden">
      {/* Wrapper box to control icon size and cropping */}
      <Box
        w="auto"
        h="100px" // Limit height to control vertical space
        overflow="hidden" // Hide excess whitespace
        display="flex"
        alignItems="center"
        justifyContent="center"
        p="0" 
        mt="-10px" // Negative margin to further reduce whitespace
        mb="-10px" // Negative margin for bottom as well
      >
        <BirdHouseIcon h="200px" w="auto" color={logoColor} transform="scale(1.2)" /> {/* Apply scaling */}
      </Box>
      <HSeparator mb="20px" />
    </Flex>
  );
}

export default SidebarBrand;
