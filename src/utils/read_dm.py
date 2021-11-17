###############################################################################
#
# read_dm.py
#
# This is used to read DigitalMicrograph data (.dm3 and .dm4)
# copied from sources below
#
# I may have modified this slightly, but frankly I have forgotten
#
###############################################################################

###############################################################################
# Python script for parsing GATAN DM3 (DigitalMicrograph) files
# --
# warning: *tested on single-image files only*
# --
# based on the DM3_Reader plug-in (v 1.3.4) for ImageJ
# by Greg Jefferis <jefferis@stanford.edu>
# http://rsb.info.nih.gov/ij/plugins/DM3_Reader.html
# --
# Python adaptation: Pierre-Ivan Raynal <raynal@univ-tours.fr>
# http://microscopies.med.univ-tours.fr/
###############################################################################

import os
import time
import struct
import numpy as np
from PIL import Image

__all__ = ["DM3", "VERSION"]

VERSION = '1.0'

debugLevel = 0   # 0=none, 1-3=basic, 4-5=simple, 6-10 verbose


### utility fuctions ###
# Image to Array
def im2ar(image_):
    """Convert PIL Image to Numpy array."""
    if image_.mode in ('L', 'I', 'F'):
        # Warning: only works with PIL.Image.Image
        # whose mode is 'L', 'I' or 'F'
        #          => error if mode == 'I;16' for instance
        array_ = np.asarray(image_)
        return array_
#    else:
#        return False

### binary data reading functions ###

def readLong(f):
    """Read 4 bytes as integer in file f"""
    read_bytes = f.read(4)
    return struct.unpack('>l', read_bytes)[0]


def readShort(f):
    """Read 2 bytes as integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('>h', read_bytes)[0]


def readByte(f):
    """Read 1 byte as integer in file f"""
    read_bytes = f.read(1)
    return struct.unpack('>b', read_bytes)[0]


def readBool(f):
    """Read 1 byte as boolean in file f"""
    read_val = readByte(f)
    return (read_val != 0)


def readChar(f):
    """Read 1 byte as char in file f"""
    read_bytes = f.read(1)
    return struct.unpack('c', read_bytes)[0]


def readString(f, len_=1):
    """Read len_ bytes as a string in file f"""
    read_bytes = f.read(len_)
    str_fmt = '>' + str(len_) + 's'
    return struct.unpack(str_fmt, read_bytes)[0]


def readLEShort(f):
    """Read 2 bytes as *little endian* integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('<h', read_bytes)[0]


def readLELong(f):
    """Read 4 bytes as *little endian* integer in file f"""
    read_bytes = f.read(4)
    return struct.unpack('<l', read_bytes)[0]


def readLEUShort(f):
    """Read 2 bytes as *little endian* unsigned integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('<H', read_bytes)[0]


def readLEULong(f):
    """Read 4 bytes as *little endian* unsigned integer in file f"""
    read_bytes = f.read(4)
    return struct.unpack('<L', read_bytes)[0]


def readLEFloat(f):
    """Read 4 bytes as *little endian* float in file f"""
    read_bytes = f.read(4)
    return struct.unpack('<f', read_bytes)[0]


def readLEDouble(f):
    """Read 8 bytes as *little endian* double in file f"""
    read_bytes = f.read(8)
    return struct.unpack('<d', read_bytes)[0]


def readLELongLong(f):
    """Read 8 bytes as *little endian* integer in file f"""
    read_bytes = f.read(8)
    return struct.unpack('<q', read_bytes)[0]


def readLEULongLong(f):
    """Read 8 bytes as *little endian* unsigned integer in file f"""
    read_bytes = f.read(8)
    return struct.unpack('<Q', read_bytes)[0]

## constants for encoded data types ##
SHORT = 2
LONG = 3
USHORT = 4
ULONG = 5
FLOAT = 6
DOUBLE = 7
BOOLEAN = 8
CHAR = 9
OCTET = 10
LONGLONG = 11
ULONGLONG = 12
STRUCT = 15
STRING = 18
ARRAY = 20

# - association data type <--> reading function
readFunc = {
    SHORT: readLEShort,
    LONG: readLELong,
    USHORT: readLEUShort,
    ULONG: readLEULong,
    FLOAT: readLEFloat,
    DOUBLE: readLEDouble,
    BOOLEAN: readBool,
    CHAR: readChar,
    OCTET: readChar,    # difference with char???
    LONGLONG: readLELongLong,
    ULONGLONG: readLEULongLong,
}

## list of image DataTypes ##
dataTypes = {
    0:  'NULL_DATA',
    1:  'SIGNED_INT16_DATA',
    2:  'REAL4_DATA',
    3:  'COMPLEX8_DATA',
    4:  'OBSELETE_DATA',
    5:  'PACKED_DATA',
    6:  'UNSIGNED_INT8_DATA',
    7:  'SIGNED_INT32_DATA',
    8:  'RGB_DATA',
    9:  'SIGNED_INT8_DATA',
    10: 'UNSIGNED_INT16_DATA',
    11: 'UNSIGNED_INT32_DATA',
    12: 'REAL8_DATA',
    13: 'COMPLEX16_DATA',
    14: 'BINARY_DATA',
    15: 'RGB_UINT8_0_DATA',
    16: 'RGB_UINT8_1_DATA',
    17: 'RGB_UINT16_DATA',
    18: 'RGB_FLOAT32_DATA',
    19: 'RGB_FLOAT64_DATA',
    20: 'RGBA_UINT8_0_DATA',
    21: 'RGBA_UINT8_1_DATA',
    22: 'RGBA_UINT8_2_DATA',
    23: 'RGBA_UINT8_3_DATA',
    24: 'RGBA_UINT16_DATA',
    25: 'RGBA_FLOAT32_DATA',
    26: 'RGBA_FLOAT64_DATA',
    27: 'POINT2_SINT16_0_DATA',
    28: 'POINT2_SINT16_1_DATA',
    29: 'POINT2_SINT32_0_DATA',
    30: 'POINT2_FLOAT32_0_DATA',
    31: 'RECT_SINT16_1_DATA',
    32: 'RECT_SINT32_1_DATA',
    33: 'RECT_FLOAT32_1_DATA',
    34: 'RECT_FLOAT32_0_DATA',
    35: 'SIGNED_INT64_DATA',
    36: 'UNSIGNED_INT64_DATA',
    37: 'LAST_DATA',
}

## other constants ##
IMGLIST = "root.ImageList."
OBJLIST = "root.DocumentObjectList."
MAXDEPTH = 64

DEFAULTCHARSET = 'utf-8'
## END constants ##


class DM3(object):

    """DM3 object. """

    # Version specific integer reading
    def readLEVersionInt(self, f):
        """Read n bytes as *little endian* unsigned integer in file f, size depending on file version"""
        if self.fileVersion == 3:
            read_bytes = f.read(4)
            return struct.unpack('>L', read_bytes)[0]
        elif self.fileVersion == 4:
            read_bytes = f.read(8)
            return struct.unpack('>Q', read_bytes)[0]
        else:
            raise Exception("Cannot process none DM3/DM4 files")

    # utility functions
    def _makeGroupString(self):
        tString = self._curGroupAtLevelX[0]
        for i in range(1, self._curGroupLevel + 1):
            tString += '.' + self._curGroupAtLevelX[i]
        return tString

    def _makeGroupNameString(self):
        tString = self._curGroupNameAtLevelX[0]
        for i in range(1, self._curGroupLevel + 1):
            tString += '.' + str(self._curGroupNameAtLevelX[i])
        return tString

    def _readTagGroup(self):
        # go down a level
        self._curGroupLevel += 1
        # increment group counte
        self._curGroupAtLevelX[self._curGroupLevel] += 1
        # set number of current tag to -1
        # --- readTagEntry() pre-increments => first gets 0
        self._curTagAtLevelX[self._curGroupLevel] = -1
        if (debugLevel > 5):
            print("rTG: Current Group Level:", self._curGroupLevel)
        # is the group sorted?
        sorted_ = readByte(self._f)
        isSorted = (sorted_ == 1)
        # is the group open?
        opened = readByte(self._f)
        isOpen = (opened == 1)
        # number of Tags
        nTags = self.readLEVersionInt(self._f)
        if (debugLevel > 5):
            print("rTG: Iterating over the", nTags,
                  "tag entries in this group")
        # read Tags
        for i in range(nTags):
            self._readTagEntry()
        # go back up one level as reading group is finished
        self._curGroupLevel += -1
        return 1

    def _readTagEntry(self):
        # is data or a new group?
        data = readByte(self._f)
        isData = (data == 21)
        self._curTagAtLevelX[self._curGroupLevel] += 1
        # get tag label if exists
        lenTagLabel = readShort(self._f)
        if (lenTagLabel != 0):
            tagLabel = readString(self._f, lenTagLabel).decode("latin-1")
        else:
            tagLabel = str(self._curTagAtLevelX[self._curGroupLevel])
        if (debugLevel > 5):
            print(
                str(self._curGroupLevel) + "|" + self._makeGroupString() + ":")
            print("Tag label = " + tagLabel)
        elif (debugLevel > 0):
            print(str(self._curGroupLevel) + ": Tag label = " + tagLabel)

        # read pointless(?) dm4 uint64
        if self.fileVersion == 4:
            unused = readLEULongLong(self._f)

        if isData:
            # give it a name
            self._curTagName = self._makeGroupNameString() + "." + tagLabel
            # read it
            self._readTagType()
        else:
            # it is a tag group
            self._curGroupNameAtLevelX[self._curGroupLevel + 1] = tagLabel
            self._readTagGroup()  # increments curGroupLevel
        return 1

    def _readTagType(self):
        delim = readString(self._f, 4).decode("utf-8")
        if (delim != '%%%%'):
            raise Exception(hex(self._f.tell())
                            + ": Tag Type delimiter not %%%%")
        nInTag = self.readLEVersionInt(self._f)
        self._readAnyData()
        return 1

    def _encodedTypeSize(self, eT):
        # returns the size in bytes of the data type
        if eT == 0:
            width = 0
        elif eT in (BOOLEAN, CHAR, OCTET):
            width = 1
        elif eT in (SHORT, USHORT):
            width = 2
        elif eT in (LONG, ULONG, FLOAT):
            width = 4
        elif eT in (DOUBLE, LONGLONG, ULONGLONG):
            width = 8
        else:
            # returns -1 for unrecognised types
            width = -1
        return width

    def _readAnyData(self):
        # higher level function dispatching to handling data types
        # to other functions
        # - get Type category (short, long, array...)
        encodedType = self.readLEVersionInt(self._f)
        # - calc size of encodedType
        etSize = self._encodedTypeSize(encodedType)
        if (debugLevel > 5):
            print("rAnD, " + hex(self._f.tell()) + ":", end=' ')
            print("Tag Type = " + str(encodedType) + ",", end=' ')
            print("Tag Size = " + str(etSize))
        if (etSize > 0):
            self._storeTag(self._curTagName,
                           self._readNativeData(encodedType, etSize))
        elif (encodedType == STRING):
            stringSize = readLong(self._f)
            self._readStringData(stringSize)
        elif (encodedType == STRUCT):
            # does not store tags yet
            structTypes = self._readStructTypes()
            self._readStructData(structTypes)
        elif (encodedType == ARRAY):
            # does not store tags yet
            # indicates size of skipped data blocks
            arrayTypes = self._readArrayTypes()
            self._readArrayData(arrayTypes)
        else:
            print(encodedType)
            print("\n\n\n\n\n\n\n")
            raise Exception("rAnD, " + hex(self._f.tell())
                            + ": Can't understand encoded type")
        return 1

    def _readNativeData(self, encodedType, etSize):
        # reads ordinary data types
        if encodedType in readFunc:
            val = readFunc[encodedType](self._f)
        else:
            raise Exception("rND, " + hex(self._f.tell())
                            + ": Unknown data type " + str(encodedType))
        if (debugLevel > 3):
            print("rND, " + hex(self._f.tell()) + ": " + str(val))
        elif (debugLevel > 0):
            print(val)
        return val

    def _readStringData(self, stringSize):
        # reads string data
        if (stringSize <= 0):
            rString = ""
        else:
            if (debugLevel > 3):
                print("rSD @ " + str(self._f.tell()) + "/" +
                      hex(self._f.tell()) + " :", end=' ')
            # !!! *Unicode* string (UTF-16)... convert to Python unicode str
            rString = readString(self._f, stringSize)
            rString = str(rString, "utf_16_le")
            if (debugLevel > 3):
                print(rString + "   <" + repr(rString) + ">")
        if (debugLevel > 0):
            print("StringVal:", rString)
        self._storeTag(self._curTagName, rString)
        return rString

    def _readArrayTypes(self):
        # determines the data types in an array data type
        arrayType = self.readLEVersionInt(self._f)
        itemTypes = []
        if (arrayType == STRUCT):
            itemTypes = self._readStructTypes()
        elif (arrayType == ARRAY):
            itemTypes = self._readArrayTypes()
        else:
            itemTypes.append(arrayType)
        return itemTypes

    def _readArrayData(self, arrayTypes):
        # reads array data

        arraySize = self.readLEVersionInt(self._f)

        if (debugLevel > 3):
            print("rArD, " + hex(self._f.tell()) + ":", end=' ')
            print("Reading array of size = " + str(arraySize))

        itemSize = 0
        encodedType = 0

        for i in range(len(arrayTypes)):
            encodedType = int(arrayTypes[i])
            etSize = self._encodedTypeSize(encodedType)
            itemSize += etSize
            if (debugLevel > 5):
                print("rArD: Tag Type = " + str(encodedType) + ",", end=' ')
                print("Tag Size = " + str(etSize))
            ##! readNativeData( encodedType, etSize ) !##

        if (debugLevel > 5):
            print("rArD: Array Item Size = " + str(itemSize))

        bufSize = arraySize * itemSize

        if ((not self._curTagName.endswith("ImageData.Data"))
                and (len(arrayTypes) == 1)
                and (encodedType == USHORT)
                and (arraySize < 256)):
            # treat as string
            val = self._readStringData(bufSize)
        else:
            # treat as binary data
            # - store data size and offset as tags
            self._storeTag(self._curTagName + ".Size", bufSize)
            self._storeTag(self._curTagName + ".Offset", self._f.tell())
            # - skip data w/o reading
            self._f.seek(self._f.tell() + bufSize)

        return 1

    def _readStructTypes(self):
        # analyses data types in a struct

        if (debugLevel > 3):
            print("Reading Struct Types at Pos = " + hex(self._f.tell()))

        structNameLength = self.readLEVersionInt(self._f)
        nFields = self.readLEVersionInt(self._f)

        if (debugLevel > 5):
            print("nFields = ", nFields)

        if (nFields > 100):
            raise Exception(hex(self._f.tell()) + ": Too many fields")

        fieldTypes = []
        nameLength = 0
        for i in range(nFields):
            nameLength = self.readLEVersionInt(self._f)
            if (debugLevel > 9):
                print(i + "th namelength = " + nameLength)
            fieldType = self.readLEVersionInt(self._f)
            fieldTypes.append(fieldType)

        return fieldTypes

    def _readStructData(self, structTypes):
        # reads struct data based on type info in structType
        for i in range(len(structTypes)):
            encodedType = structTypes[i]
            etSize = self._encodedTypeSize(encodedType)

            if (debugLevel > 5):
                print("Tag Type = " + str(encodedType) + ",", end=' ')
                print("Tag Size = " + str(etSize))

            # get data
            self._readNativeData(encodedType, etSize)

        return 1

    def _storeTag(self, tagName, tagValue):
        # NB: all tag values (and names) stored as unicode objects;
        #     => can then be easily converted to any encoding
        # - /!\ tag names may not be ascii char only
        #       (e.g. '\xb5', i.e. MICRO SIGN)

        # tagName = str(tagName, 'latin-1')

        # - convert tag value to unicode if not already unicode object
        #   (as for string data)
        tagValue = str(tagValue)
        # store Tags as list and dict
        self._storedTags.append(tagName + " = " + tagValue)
        self._tagDict[tagName] = tagValue

    ### END utility functions ###

    def __init__(self, filename, debug=0):
        """DM3 object: parses DM3 file."""

        ## initialize variables ##
        self.debug = debug
        self._outputcharset = DEFAULTCHARSET
        self._filename = filename
        self._chosenImage = 1
        isDM3 = False
        isDM4 = False
        # - track currently read group
        self._curGroupLevel = -1
        self._curGroupAtLevelX = [0 for x in range(MAXDEPTH)]
        self._curGroupNameAtLevelX = ['' for x in range(MAXDEPTH)]
        # - track current tag
        self._curTagAtLevelX = ['' for x in range(MAXDEPTH)]
        self._curTagName = ''
        # - open file for reading
        self._f = open(self._filename, 'rb')
        # - create Tags repositories
        self._storedTags = []
        self._tagDict = {}

        if self.debug > 0:
            t1 = time.time()
        # read header (first 3 4-byte int)
        # get version

        self.fileVersion = readLong(self._f)
        # check for version
        if (self.fileVersion == 3):
            isDM3 = True
        if (self.fileVersion == 4):
            isDM4 = True

        # get indicated file size
        fileSize = self.readLEVersionInt(self._f)
        # get byte-ordering
        lE = readLong(self._f)
        littleEndian = (lE == 1)

        if not littleEndian:
            isDM3 = False
            isDM4 = False
        # check file header, raise Exception if not DM3
        if not isDM3 and not isDM4:
            raise Exception("%s does not appear to be a DM3 or DM4 file."
                            % os.path.split(self._filename)[1])
        elif self.debug > 0:
            if isDM3:
                print("%s appears to be a DM3 file" % (self._filename))
            elif isDM4:
                print("%s appears to be a DM4 file" % (self._filename))

        if (debugLevel > 5 or self.debug > 1):
            print("Header info.:")
            print("- file version:", self.fileVersion)
            print("- lE:", lE)
            print("- file size:", fileSize, "bytes")

        # set name of root group (contains all data)...
        self._curGroupNameAtLevelX[0] = "root"
        # ... then read it
        self._readTagGroup()
        if self.debug > 0:
            print("-- %s Tags read --" % len(self._storedTags))

        if self.debug > 0:
            t2 = time.time()
            print("| parse DM3 file: %.3g s" % (t2 - t1))

    @property
    def outputcharset(self):
        """Returns Tag dump/output charset."""
        return self._outputcharset

    @outputcharset.setter
    def outputcharset(self, value):
        """Set Tag dump/output charset."""
        self._outputcharset = value

    @property
    def filename(self):
        """Returns full file path."""
        return self._filename

    @property
    def tags(self):
        """Returns all image Tags."""
        return self._tagDict

    def dumpTags(self, dump_dir='/tmp'):
        """Dumps image Tags in a txt file."""
        dump_file = os.path.join(dump_dir,
                                 os.path.split(self._filename)[1]
                                 + ".tagdump.txt")
        try:
            dumpf = open(dump_file, 'w')
        except:
            print("Warning: cannot generate dump file.")
        else:
            for tag in self._storedTags:
                dumpf.write(tag.encode(self._outputcharset) + "\n")
            dumpf.close

    @property
    def info(self):
        """Extracts useful experiment info from DM3 file."""
        # define useful information
        tag_root = 'root.ImageList.1'
        info_keys = {
            'descrip': "%s.Description" % tag_root,
            'acq_date': "%s.ImageTags.DataBar.Acquisition Date" % tag_root,
            'acq_time': "%s.ImageTags.DataBar.Acquisition Time" % tag_root,
            'name': "%s.ImageTags.Microscope Info.Name" % tag_root,
            'micro': "%s.ImageTags.Microscope Info.Microscope" % tag_root,
            'hv': "%s.ImageTags.Microscope Info.Voltage" % tag_root,
            'mag': "%s.ImageTags.Microscope Info.Indicated Magnification" % tag_root,
            'mode': "%s.ImageTags.Microscope Info.Operation Mode" % tag_root,
            'operator': "%s.ImageTags.Microscope Info.Operator" % tag_root,
            'specimen': "%s.ImageTags.Microscope Info.Specimen" % tag_root,
            # 'image_notes': "root.DocumentObjectList.10.Text' # = Image Notes
        }
        # get experiment information
        infoDict = {}
        for key, tag_name in list(info_keys.items()):
            if tag_name in self.tags:
                # tags supplied as Python unicode str; convert to chosen charset
                # (typically latin-1 or utf-8)
                infoDict[key] = self.tags[tag_name].encode(self._outputcharset)
        # return experiment information
        return infoDict

    @property
    def thumbnail(self):
        """Returns thumbnail as PIL Image."""
        # get thumbnail
        tag_root = 'root.ImageList.0'
        tn_size = int(self.tags["%s.ImageData.Data.Size" % tag_root])
        tn_offset = int(self.tags["%s.ImageData.Data.Offset" % tag_root])
        tn_width = int(self.tags["%s.ImageData.Dimensions.0" % tag_root])
        tn_height = int(self.tags["%s.ImageData.Dimensions.1" % tag_root])

        if self.debug > 0:
            print("Notice: tn data in %s starts at %s" % (
                os.path.split(self._filename)[1], hex(tn_offset)
            ))
            print("Notice: tn size: %sx%s px" % (tn_width, tn_height))

        if (tn_width * tn_height * 4) != tn_size:
            raise Exception("Cannot extract thumbnail from %s"
                            % os.path.split(self._filename)[1])
        else:
            self._f.seek(tn_offset)
            rawdata = self._f.read(tn_size)
            # - read as 16-bit LE unsigned integer
            tn = Image.fromstring('F', (tn_width, tn_height), rawdata,
                                  'raw', 'F;32')
            # - rescale and convert px data
            tn = tn.point(lambda x: x * (1. / 65536) + 0)
            tn = tn.convert('L')
        # - return image
        return tn

    @property
    def thumbnaildata(self):
        """Returns thumbnail data as numpy.array"""
        return im2ar(self.thumbnail)

    def makePNGThumbnail(self, tn_file=''):
        """Save thumbnail as PNG file."""
        # - cleanup name
        if tn_file == '':
            tn_path = os.path.join('./',
                                   os.path.split(self.filename)[1] + '.tn.png')
        else:
            if os.path.splitext(tn_file)[1] != '.png':
                tn_path = os.path.splitext(tn_file)[0] + '.png'
            else:
                tn_path = tn_file
        # - save tn file
        try:
            self.thumbnail.save(tn_path, 'PNG')
            if self.debug > 0:
                print("Thumbnail saved as '%s'." % tn_path)
        except:
            print("Warning: could not save thumbnail.")

    @property
    def image(self):
        """Extracts image data as PIL Image"""

        # PIL "raw" decoder modes for the various image dataTypes
        dataTypesDec = {
            1: 'F;16S',  # 16-bit LE signed integer
            2: 'F;32F',  # 32-bit LE floating point
            6: 'F;8',  # 8-bit unsigned integer
            7: 'F;32S',  # 32-bit LE signed integer
            9: 'F;8S',  # 8-bit signed integer
            10: 'F;16',  # 16-bit LE unsigned integer
            11: 'F;32',  # 32-bit LE unsigned integer
            12: 'F;64F',  # 32-bit LE unsigned integer
            14: 'F;8',  # binary
        }

        npTypes = {
            1: 'int16',  # 16-bit LE signed integer
            2: 'float32',  # 32-bit LE floating point
            6: 'uint8',  # 8-bit unsigned integer
            7: 'int32',  # 32-bit LE signed integer
            9: 'int8',  # 8-bit signed integer
            10: 'uint16',  # 16-bit LE unsigned integer
            11: 'uint32',  # 32-bit LE unsigned integer
            12: 'float64',  # 32-bit LE unsigned integer
            14: 'bool',  # binary
        }

        # get relevant Tags
        tag_root = 'root.ImageList.1'
        data_offset = int(self.tags["%s.ImageData.Data.Offset" % tag_root])
        data_size = int(self.tags["%s.ImageData.Data.Size" % tag_root])
        data_type = int(self.tags["%s.ImageData.DataType" % tag_root])
        im_width = int(self.tags["%s.ImageData.Dimensions.0" % tag_root])
        im_height = int(self.tags["%s.ImageData.Dimensions.1" % tag_root])

        if "%s.ImageData.Dimensions.2" % tag_root in self.tags:
            im_depth = int(self.tags["%s.ImageData.Dimensions.2" % tag_root])
        else:
            im_depth = 1

        if self.debug > 0:
            print("Notice: image data in %s starts at %s" % (
                os.path.split(self._filename)[1], hex(data_offset)
            ))
            print("Notice: image size: %sx%s px" % (im_width, im_height))

        # check if image DataType is implemented, then read
        if data_type in dataTypesDec:
            decoder = dataTypesDec[data_type]
            if self.debug > 0:
                print("Notice: image data type: %s ('%s'), read as %s" % (
                    data_type, dataTypes[data_type], decoder
                ))
                t1 = time.time()
            self._f.seek(data_offset)
            rawdata = self._f.read(data_size)

            # Possibly can read ime into one huge array, (im_width, im_height*im_depth) then convert to numpy array of correct size
            # alternative numpy.frombuffer
            # alternative read seperate slices, need to offset the buffer? then conver them individually to arrays

            im = np.frombuffer(rawdata, dtype=np.dtype(npTypes[data_type]))
            if im_depth == 1:
                im = im.reshape((im_height, im_width))
            else:
                im = im.reshape((im_depth, im_height, im_width))
                im = np.rot90(np.rot90(im, axes=(0, 2)), -1)

            # im = Image.frombytes('F', (im_width, im_height), rawdata,
            #                       'raw', decoder)

            if self.debug > 0:
                t2 = time.time()
                print("| read image data: %.3g s" % (t2 - t1))
        else:
            raise Exception(
                "Cannot extract image data from %s: unimplemented DataType (%s:%s)." %
                (os.path.split(self._filename)[
                 1], data_type, dataTypes[data_type])
            )

        # # if image dataType is BINARY, binarize image
        # # (i.e., px_value>0 is True)
        # if data_type == 14:
        #     # convert Image to 'L' to apply point operation
        #     im = im.convert('L')
        #     # binarize
        #     im = im.point(lambda v: v > 0 or False)

        return im

    @property
    def imagedata(self):
        """Extracts image data as numpy.array"""
        return im2ar(self.image)

    @property
    def contrastlimits(self):
        """Returns display range (cuts)."""
        tag_root = 'root.DocumentObjectList.0'
        low = int(float(self.tags["%s.ImageDisplayInfo.LowLimit" % tag_root]))
        high = int(
            float(self.tags["%s.ImageDisplayInfo.HighLimit" % tag_root]))
        cuts = (low, high)
        return cuts

    @property
    def cuts(self):
        """Returns display range (cuts)."""
        return self.contrastlimits

    @property
    def pxsize(self):
        """Returns pixel size and unit."""
        tag_root = 'root.ImageList.1'
        pixel_size = float(
            self.tags["%s.ImageData.Calibrations.Dimension.0.Scale" % tag_root])
        unit = self.tags["%s.ImageData.Calibrations.Dimension.0.Units" %
                         tag_root]
        if unit == '\xb5m':
            unit = 'micron'
        else:
            unit = unit.encode('ascii')
        if self.debug > 0:
            print("pixel size = %s %s" % (pixel_size, unit))
        return (pixel_size, unit)


## MAIN ##
if __name__ == '__main__':
    print("DM3lib %s" % VERSION)