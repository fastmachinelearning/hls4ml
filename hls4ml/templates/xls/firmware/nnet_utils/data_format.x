pub enum DataFormat: u1 {
    CHANNELS_LAST = 0,
    CHANNELS_FIRST = 1
}

pub const CHANNELS_LAST = DataFormat::CHANNELS_LAST;
pub const CHANNELS_FIRST = DataFormat::CHANNELS_FIRST;

pub fn to_size_chans(dim_0: u32, dim_1: u32, data_format: DataFormat) -> u32[2] {
    match data_format {
        DataFormat::CHANNELS_LAST => [dim_0, dim_1],
        DataFormat::CHANNELS_FIRST => [dim_1, dim_0]
    }
}

pub fn from_size_chans(size: u32, channels: u32, data_format: DataFormat) -> u32[2] {
    match data_format {
        DataFormat::CHANNELS_LAST => [size, channels],
        DataFormat::CHANNELS_FIRST => [channels, size]
    }
}

pub fn to_height_width_chans(dim_0: u32, dim_1: u32, dim_2: u32, data_format: DataFormat) -> u32[3] {
    match data_format {
        DataFormat::CHANNELS_LAST => [dim_0, dim_1, dim_2],
        DataFormat::CHANNELS_FIRST => [dim_1, dim_2, dim_0]
    }
}

pub fn from_height_width_chans(height: u32, width: u32, channels: u32, data_format: DataFormat) -> u32[3] {
    match data_format {
        DataFormat::CHANNELS_LAST => [height, width, channels],
        DataFormat::CHANNELS_FIRST => [channels, height, width]
    }
}

#[test]
fn test_data_format() {
    let size = u32:4;
    let height = u32:1;
    let width = u32:2;
    let channels = u32:3;
    for (data_format, _) in [CHANNELS_LAST, CHANNELS_FIRST] {
        let size_chans = from_size_chans(size, channels, data_format);
        assert_eq(
            to_size_chans(size_chans[0], size_chans[1], data_format),
            [size, channels]
        );

        let height_width_chans = from_height_width_chans(height, width, channels, data_format);
        assert_eq(
            to_height_width_chans(
                height_width_chans[0],
                height_width_chans[1],
                height_width_chans[2],
                data_format
            ),
            [height, width, channels]
        );
    }(())
}
