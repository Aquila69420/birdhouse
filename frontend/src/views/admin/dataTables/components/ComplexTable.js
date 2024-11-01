/* eslint-disable */

import {
  Box,
  Flex,
  Icon,
  Progress,
  Table,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  useColorModeValue,
  Collapse,
} from '@chakra-ui/react';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { MdCancel, MdCheckCircle, MdOutlineError, MdExpandMore, MdExpandLess } from 'react-icons/md';
import Card from 'components/card/Card';
import Menu from 'components/menu/MainMenu';
import * as React from 'react';

const columnHelper = createColumnHelper();

export default function ComplexTable(props) {
  const { tableData } = props;
  const [sorting, setSorting] = React.useState([]);
  const textColor = useColorModeValue('secondaryGray.900', 'white');
  const borderColor = useColorModeValue('gray.200', 'whiteAlpha.100');
  const [expandedRows, setExpandedRows] = React.useState({});

  const toggleRowExpansion = (rowId) => {
    setExpandedRows((prevState) => ({
      ...prevState,
      [rowId]: !prevState[rowId],
    }));
  };

  const columns = [
    columnHelper.accessor('task-id', {
      id: 'task-id',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          TASK ID
        </Text>
      ),
      cell: (info) => (
        <Flex align="center">
          <Text color={textColor} fontSize="sm" fontWeight="700">
            {info.getValue()}
          </Text>
        </Flex>
      ),
    }),
    columnHelper.accessor('task-name', {
      id: 'task-name',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          TASK NAME
        </Text>
      ),
      cell: (info) => (
        <Flex align="center">
          <Text color={textColor} fontSize="sm" fontWeight="700">
            {info.getValue()}
          </Text>
        </Flex>
      ),
    }),
    columnHelper.accessor('task-status', {
      id: 'task-status',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          TASK STATUS
        </Text>
      ),
      cell: (info) => (
        <Flex align="center">
          <Icon
            w="24px"
            h="24px"
            me="5px"
            color={
              info.getValue() === 'Finalized'
                ? 'green.500'
                : info.getValue() === 'Disable'
                ? 'red.500'
                : info.getValue() === 'Submission'
                ? 'orange.500'
                : null
            }
            as={
              info.getValue() === 'Finalized'
                ? MdCheckCircle
                : info.getValue() === 'Disable'
                ? MdCancel
                : info.getValue() === 'Submission'
                ? MdOutlineError
                : null
            }
          />
          <Text color={textColor} fontSize="sm" fontWeight="700">
            {info.getValue()}
          </Text>
        </Flex>
      ),
    }),
    columnHelper.accessor('date', {
      id: 'date',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          DATE
        </Text>
      ),
      cell: (info) => (
        <Text color={textColor} fontSize="sm" fontWeight="700">
          {info.getValue()}
        </Text>
      ),
    }),
    columnHelper.accessor('progress', {
      id: 'progress',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          PROGRESS
        </Text>
      ),
      cell: (info) => (
        <Flex align="center">
          <Progress
            variant="table"
            colorScheme="brandScheme"
            h="8px"
            w="108px"
            value={info.getValue()}
          />
        </Flex>
      ),
    }),
    columnHelper.accessor('percentage', {
      id: 'percentage',
      header: () => (
        <Text fontSize={{ sm: '10px', lg: '12px' }} color="gray.400">
          % IN DAILY REWARDS POOL
        </Text>
      ),
      cell: (info) => (
        <Flex align="center">
          <Progress
            variant="table"
            colorScheme="brandScheme"
            h="8px"
            w="108px"
            value={info.getValue()}
          />
        </Flex>
      ),
    }),
    columnHelper.accessor('expand', {
      id: 'expand',
      header: () => null, // No header for the expand icon column
      cell: (info) => (
        <Icon
          as={expandedRows[info.row.id] ? MdExpandLess : MdExpandMore}
          w="20px"
          h="20px"
          cursor="pointer"
          onClick={(e) => {
            e.stopPropagation(); // Prevents row click from firing twice
            toggleRowExpansion(info.row.id);
          }}
        />
      ),
    }),
  ];

  const table = useReactTable({
    data: tableData,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    debugTable: true,
  });

  return (
    <Card flexDirection="column" w="100%" px="0px" overflowX="auto">
      <Flex px="25px" mb="8px" justifyContent="space-between" align="center">
        <Text color={textColor} fontSize="22px" fontWeight="700">
          My Tasks
        </Text>
        <Menu />
      </Flex>
      <Box>
        <Table variant="simple" color="gray.500" mb="24px" mt="12px">
          <Thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <Tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <Th
                    key={header.id}
                    borderColor={borderColor}
                    cursor="pointer"
                    onClick={header.column.getToggleSortingHandler()}
                    _hover={{ cursor: 'pointer' }}
                  >
                    <Flex justifyContent="space-between" align="center">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                    </Flex>
                  </Th>
                ))}
              </Tr>
            ))}
          </Thead>
          <Tbody>
            {table.getRowModel().rows.map((row) => (
              <React.Fragment key={row.id}>
                <Tr
                  onClick={() => toggleRowExpansion(row.id)}
                  cursor="pointer"
                  _hover={{ bg: 'gray.100' }} // Adds a subtle background change on hover
                >
                  {row.getVisibleCells().map((cell) => (
                    <Td key={cell.id} borderColor="transparent">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </Td>
                  ))}
                </Tr>
                <Tr>
                  <Td colSpan={columns.length} p="0">
                    <Collapse in={expandedRows[row.id]} animateOpacity>
                      <Box p="20px" bg="gray.50" borderBottomRadius="md">
                        <Text fontSize="sm" color="gray.600">
                          {row.original.description}
                        </Text>
                      </Box>
                    </Collapse>
                  </Td>
                </Tr>
              </React.Fragment>
            ))}
          </Tbody>
        </Table>
      </Box>
    </Card>
  );
}
