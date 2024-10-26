// Chakra imports
import { Box, SimpleGrid, Input, Button, Text, Flex } from "@chakra-ui/react";
import ComplexTable from "views/admin/dataTables/components/ComplexTable-Validate";
import PastTasks from "views/admin/dataTables/components/PastTasks"; // Import the PastTasks component
import { columnsDataComplex } from "views/admin/dataTables/variables/columnsData";
import tableDataComplex from "views/admin/dataTables/variables/tableDataComplex.json";
import React, { useState } from "react";

export default function Validate() {
  const [selectedTask, setSelectedTask] = useState(null);
  const [tokens, setTokens] = useState("");

  // Handle staking button click
  const handleStake = () => {
    console.log("Stake Action:", { task: selectedTask, tokens });
    setSelectedTask(null);
    setTokens("");
  };

  return (
    <Box pt={{ base: "130px", md: "80px", xl: "80px" }} h="100vh" overflow="auto">
      <SimpleGrid columns={{ base: 1, md: 5 }} spacing="20px" h="100%">
        {/* Main Table - Takes up 80% of the grid */}
        <Box gridColumn={{ base: "span 1", md: "span 4" }}>
          <ComplexTable
            columnsData={columnsDataComplex}
            tableData={tableDataComplex}
            onTaskSelect={setSelectedTask} // Pass selected task to show staking field
          />

          {selectedTask && (
            <Flex direction="column" mt="4" align="center" w="100%">
              <Text fontSize="lg" fontWeight="600" mb="2">
                Stake Tokens for Selected Task
              </Text>
              <Input
                placeholder="Enter tokens amount"
                type="number"
                value={tokens}
                onChange={(e) => setTokens(e.target.value)}
                width="300px"
                mb="4"
              />
              <Button
                colorScheme="brandScheme"
                bg="brand.500"
                color="white"
                onClick={handleStake}
                _hover={{ bg: "brand.600" }}
                _active={{ bg: "brand.700" }}
              >
                Stake FML
              </Button>
            </Flex>
          )}
        </Box>

        {/* Past Validated Tasks - Takes up 20% of the grid */}
        <Box gridColumn={{ base: "span 1", md: "span 1" }}>
          <PastTasks /> {/* Displays past validated tasks */}
        </Box>
      </SimpleGrid>
    </Box>
  );
}
