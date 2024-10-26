// Chakra and React Imports
import { Box, Button, Checkbox, CheckboxGroup, SimpleGrid, Stack, Text } from "@chakra-ui/react";
import { useState } from "react";
import axios from "axios";

// Import your tables and data
import DevelopmentTable from "views/admin/dataTables/components/DevelopmentTable";
import CheckTable from "views/admin/dataTables/components/CheckTable";
import ColumnsTable from "views/admin/dataTables/components/ColumnsTable";
import ComplexTable from "views/admin/dataTables/components/ComplexTable";
import {
  columnsDataDevelopment,
  columnsDataCheck,
  columnsDataColumns,
  columnsDataComplex,
} from "views/admin/dataTables/variables/columnsData";
import tableDataDevelopment from "views/admin/dataTables/variables/tableDataDevelopment.json";
import tableDataCheck from "views/admin/dataTables/variables/tableDataCheck.json";
import tableDataColumns from "views/admin/dataTables/variables/tableDataColumns.json";
import tableDataComplex from "views/admin/dataTables/variables/tableDataComplex.json";

export default function Training() {
  const [selectedItems, setSelectedItems] = useState([]);

  // Function to handle API call
  const handleTrain = async () => {
    try {
      const response = await axios.post("/api/train", { items: selectedItems });
      console.log("Training response:", response.data);
      // Handle any response or feedback here
    } catch (error) {
      console.error("Error during training:", error);
    }
  };

  return (
    <Box pt={{ base: "130px", md: "80px", xl: "80px" }}>
      <SimpleGrid mb="20px" columns={{ sm: 1, md: 2 }} spacing={{ base: "20px", xl: "20px" }}>
        <DevelopmentTable columnsData={columnsDataDevelopment} tableData={tableDataDevelopment} />
        <CheckTable columnsData={columnsDataCheck} tableData={tableDataCheck} />
        <ColumnsTable columnsData={columnsDataColumns} tableData={tableDataColumns} />
        <ComplexTable columnsData={columnsDataComplex} tableData={tableDataComplex} />
      </SimpleGrid>

      {/* Training Button and Checklist */}
      <Box mb="20px">
        <Text fontSize="lg" fontWeight="bold" mb="10px">
          Select Training Options:
        </Text>
        <CheckboxGroup
          colorScheme="green"
          onChange={setSelectedItems}
          value={selectedItems}
        >
          <Stack spacing={2}>
            <Checkbox value="Option 1">Option 1</Checkbox>
            <Checkbox value="Option 2">Option 2</Checkbox>
            <Checkbox value="Option 3">Option 3</Checkbox>
            {/* Add more options as needed */}
          </Stack>
        </CheckboxGroup>

        <Button
          mt="20px"
          colorScheme="teal"
          onClick={handleTrain}
          isDisabled={selectedItems.length === 0}
        >
          Start Training
        </Button>
      </Box>
    </Box>
  );
}
