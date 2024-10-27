// Training.jsx

import React, { useState } from "react";
import axios from "axios";
import { Box, Stack, Flex, Input, Button, Text } from "@chakra-ui/react";
import TrainingTasks from "views/admin/dataTables/components/TrainingTasks";
import PastTrainingTasks from "views/admin/dataTables/components/PastTrainingTasks";

export default function Training() {
  const [selectedTask, setSelectedTask] = useState(null);
  const [tokens, setTokens] = useState("");

  const handleStake = () => {
    console.log("Stake Action:", { task: selectedTask, tokens });
    setSelectedTask(selectedTask);
    let token_updated;
    // TODO: Update the wallet id
    axios.post('http://10.154.36.81:5000/pay_tokens', {
      wallet_id: 'CNN',
      tokens: tokens
    }).then((res) => {
      token_updated = res.data;
    });
    setTokens(token_updated);
  };

  return (
    <Box pt={{ base: "130px", md: "80px", xl: "80px" }} h="100vh" overflow="auto">
      <Stack spacing="20px">
        
        {/* Current Training Tasks Table */}
        <Box>
          <TrainingTasks onTaskSelect={setSelectedTask} />
          
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

        {/* Past Training Tasks Section */}
        <Box>
          <PastTrainingTasks />
        </Box>
      </Stack>
    </Box>
  );
}
