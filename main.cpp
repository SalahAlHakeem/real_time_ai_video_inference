#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <map>
#include <chrono>

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gmodule.h>
#include <cuda_runtime.h>

#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"
#include "nvds_version.h"
#include "nvdsmeta.h"
#include "nvdsinfer.h"

#include "nvdsinfer_custom_impl.h"
#include "gstnvdsinfer.h"

#include "cuda_runtime_api.h"
#include "cv/core/Tensor.h"
#include "nvbufsurface.h"

#include "nvds_yml_parser.h"
#include "ds_yml_parse.h"
#include <yaml-cpp/yaml.h>


#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvds_obj_encode.h"
#include <nvds_obj_encode.h>

#include <amqp.h>
#include <amqp_tcp_socket.h>
#include <amqp_framing.h>

#include <nlohmann/json.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <cryptopp/base64.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include <yaml-cpp/yaml.h>

#include "./common/ds_yml_parse.h"

using namespace std;

#define PGIE_CLASS_ID_FACE 0

#define PGIE_DETECTED_CLASS_NUM 4

unordered_set<guint64> registered_tracking_objects;
unordered_set<guint64> message_broker_sent_item_ids;
const vector<int> SUSPICIOUS_ITEM_IDS {24, 26, 28};


/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
#define CONFIG_GPU_ID "gpu-id"

#define SGIE_NET_WIDTH 80
#define SGIE_NET_HEIGHT 80

gint frame_number = 0;

#define PRIMARY_DETECTOR_UID 1

GstElement *pipeline = NULL;

#define GENERAL_CONFIG "configurations/app_general_config.yml"
#define PRIMARY_ENGINE_CONFIG "configurations/primary_engine_config.yml"
#define NVDS_TRACKER_CONFIG "configurations/tracker_config.yml"

YAML::Node app_config = YAML::LoadFile("configurations/app_general_config.yml");

const char* AMQP_EXCHANGE = app_config["message-broker"]["exchange"].as<std::string>().c_str();
const char* AMQP_ROUTING_KEY = app_config["message-broker"]["routing-key"].as<std::string>().c_str();

struct BBox {
    float left;
    float top;
    float width;
    float height;

    BBox(): left(0), top(0), width(0), height(0) {}

    BBox(float left, float top, float width, float height) {
        this->left = left;
        this->top = top;
        this->width = width;
        this->height = height;
    }
};

struct DetectedItem {
    int item_category_id;
    guint64 item_tracking_id;
    BBox item_bbox;
    std::time_t now_c;

    DetectedItem(int item_category_id, guint64 item_tracking_id, BBox item_bbox) {
        this->item_category_id = item_category_id;
        this->item_tracking_id = item_tracking_id;
        this->item_bbox = item_bbox;

        auto now = std::chrono::system_clock::now();
        this->now_c = std::chrono::system_clock::to_time_t(now);
    }
};

vector<DetectedItem> deteceted_items_in_track;
unordered_set<guint64> reported_left_items;

typedef struct _perf_measure{
    GstClockTime pre_time;
    GstClockTime total_time;
    guint count;
}perf_measure;

typedef struct _DsSourceBin
{
    GstElement *source_bin;
    GstElement *uri_decode_bin;
    GstElement *vidconv;
    GstElement *nvvidconv;
    GstElement *capsfilt;
    gint index;
}DsSourceBinStruct;

typedef struct {
    amqp_connection_state_t* rabbitmq_connection;
    NvDsObjEncCtxHandle ctx_handle;
    BBox* detected_item_bbox;
    guint64 detected_item_id;
} AppData;


static void signal_catch_callback(int signum) {
    g_print("User Interrupted..\n");
    if (pipeline != NULL) {
        gst_element_send_event(pipeline, gst_event_new_eos());
        g_print("Send EOS to pipeline!\n");
    }
}

static void signal_catch_setup() {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = signal_catch_callback;
    sigaction(SIGINT, &action, NULL);
}

// Function to check if two bounding boxes overlap
bool isOverlap(const BBox &a, const BBox &b) {
    return (a.left < b.left + b.width &&
            a.left + a.width > b.left &&
            a.top < b.top + b.height &&
            a.top + a.height > b.top);
}

std::string base64Encode(const std::vector<unsigned char>& data) {
    std::string encoded;
    CryptoPP::Base64Encoder encoder;
    CryptoPP::StringSink* stringSink = new CryptoPP::StringSink(encoded);
    encoder.Attach(stringSink);
    encoder.Put(data.data(), data.size());
    encoder.MessageEnd();
    return encoded;
}

std::vector<unsigned char> readFileToBytes(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
    return buffer;
}


/**
 * @brief Callback function to handle messages on the GStreamer bus.
 *
 * This function is called whenever a message is posted on the GStreamer bus.
 * It handles end-of-stream (EOS) and error messages. For EOS, it stops the
 * main loop, effectively stopping the pipeline. For error messages, it
 * prints the error details and stops the main loop.
 *
 * @param bus The GStreamer bus from which the message was received.
 * @param msg The message that was posted on the bus.
 * @param data User data passed to the callback function, typically the main loop.
 * @return TRUE to indicate that the message was handled.
 */
static gboolean bus_call (GstBus * bus, GstMessage * msg, gpointer data) {
    // Cast the user data to a GMainLoop pointer.
    GMainLoop *loop = (GMainLoop *) data;

    // Switch based on the type of the message.
    switch (GST_MESSAGE_TYPE (msg)) {
        // Case for end-of-stream (EOS) message.
        case GST_MESSAGE_EOS:
            g_print ("End of stream\n");
            // Quit the main loop, stopping the pipeline.
            g_main_loop_quit (loop);
            break;

            // Case for error message.
        case GST_MESSAGE_ERROR: {
            gchar *debug;  // Variable to hold debug information.
            GError *error; // Variable to hold the error.

            // Parse the error message to get the error and debug information.
            gst_message_parse_error (msg, &error, &debug);

            // Print the error message and the element that generated it.
            g_printerr ("ERROR from element %s: %s\n",
                        GST_OBJECT_NAME (msg->src), error->message);

            // If debug information is available, print it.
            if (debug)
                g_printerr ("Error details: %s\n", debug);

            // Free the debug information.
            g_free (debug);

            // Free the error.
            g_error_free (error);

            // Quit the main loop, stopping the pipeline.
            g_main_loop_quit (loop);
            break;
        }

            // Default case for other message types.
        default:
            break;
    }

    // Return TRUE to indicate that the message was handled.
    return TRUE;
}


static GstPadProbeReturn pgie_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    GstClockTime now;
    perf_measure *perf = (perf_measure *)(u_data);

    // Retrieve batch metadata from the buffer
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    // Get the current monotonic time
    now = g_get_monotonic_time();

    // Initialize or update performance measurement
    if (perf->pre_time == GST_CLOCK_TIME_NONE) {
        perf->pre_time = now;
        perf->total_time = GST_CLOCK_TIME_NONE;
    } else {
        if (perf->total_time == GST_CLOCK_TIME_NONE) {
            perf->total_time = (now - perf->pre_time);
        } else {
            perf->total_time += (now - perf->pre_time);
        }
        perf->pre_time = now;
        perf->count++;
    }

    // Iterate through each frame metadata in the batch
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);


        if (!frame_meta)
            continue;

        // Iterate through each object metadata in the frame
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *)(l_obj->data);

            if (!obj_meta)
                continue;

            bool is_suspicious_category = false;

            for (int current_item: SUSPICIOUS_ITEM_IDS) {
                if (current_item == obj_meta->class_id || obj_meta->class_id == 0) {
                    is_suspicious_category = true;
                    break;
                }
            }


            if (!is_suspicious_category) {
                obj_meta->rect_params.border_width = 0.0;
                obj_meta->rect_params.border_color.alpha = 0.0;

                if (obj_meta->text_params.display_text) {
                    g_free(obj_meta->text_params.display_text);
                }

                obj_meta->text_params.display_text = g_strdup("");
                continue;
            }

            for (DetectedItem& current_detected_item: deteceted_items_in_track) {
                if (obj_meta->object_id == current_detected_item.item_tracking_id) {
                    obj_meta->text_params.display_text = g_strdup_printf("Bag");
                    obj_meta->text_params.set_bg_clr = 1;
                    obj_meta->text_params.text_bg_clr.red = 0.0;
                    obj_meta->text_params.text_bg_clr.green = 0.0;
                    obj_meta->text_params.text_bg_clr.blue = 0.0;
                    obj_meta->text_params.text_bg_clr.alpha = 1.0;

                    obj_meta->rect_params.border_color.red = 0.0;
                    obj_meta->rect_params.border_color.green = 1.0;
                    obj_meta->rect_params.border_color.blue = 0.0;
                    obj_meta->rect_params.border_color.alpha = 1.0;
                }
            }


            /* Check that the object has been detected by the primary detector
             * and that the class id is that of faces. */
            if (obj_meta->unique_component_id == PRIMARY_DETECTOR_UID) {

                auto insertion_result = registered_tracking_objects.insert(obj_meta->object_id);

                if (insertion_result.second) {
                    float left = obj_meta->rect_params.left;
                    float top = obj_meta->rect_params.top;
                    float width = obj_meta->rect_params.width;
                    float height = obj_meta->rect_params.height;

                    BBox current_item_bbox {left, top, width, height};

                    for (int currentElement: SUSPICIOUS_ITEM_IDS) {
                        if (currentElement == obj_meta->class_id) {
                            DetectedItem current_detected_item {obj_meta->class_id, obj_meta->object_id, current_item_bbox};

                            deteceted_items_in_track.push_back(current_detected_item);

                            cout << "new tracking left item with object id: " << obj_meta->object_id << endl;
                        }
                    }
                }
            }
        }
    }

    // Print the frame number and face count
    frame_number++;
    return GST_PAD_PROBE_OK;
}



static GstPadProbeReturn message_broker_probe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
    AppData* app_data = (AppData*) u_data;

    if (app_data->detected_item_id == 0)
        return GST_PAD_PROBE_OK;

    auto result = message_broker_sent_item_ids.insert(app_data->detected_item_id);

    if (result.second) {
        nlohmann::json broker_msg, item_bbox;

        std::vector<unsigned char> imageData = readFileToBytes("./suspicious_item.jpg");

        std::string base64String = base64Encode(imageData);

        item_bbox["top"] = app_data->detected_item_bbox->top;
        item_bbox["left"] = app_data->detected_item_bbox->left;
        item_bbox["width"] = app_data->detected_item_bbox->width;
        item_bbox["height"] = app_data->detected_item_bbox->height;

        broker_msg["item_bbox"] = item_bbox;
        broker_msg["frame_image"] = base64String;

        std::string message_body = broker_msg.dump();

        amqp_bytes_t message;
        message.len = message_body.length();
        message.bytes = (void*) message_body.c_str();

       amqp_basic_publish(*app_data->rabbitmq_connection, 1, amqp_cstring_bytes(AMQP_EXCHANGE), amqp_cstring_bytes(AMQP_ROUTING_KEY), 0,0, NULL, message);

    }

    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn bbox_overlap_verification_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(GST_BUFFER(info->data));

    GstBuffer *buf = (GstBuffer *) info->data;  // Retrieve the GstBuffer from the probe info
    GstMapInfo inmap = GST_MAP_INFO_INIT;

    // Map the buffer to access the data
    if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
        GST_ERROR ("input buffer mapinfo failed");
        return GST_PAD_PROBE_DROP;
    }

    // Access the NvBufSurface data from the mapped buffer
    NvBufSurface *ip_surf = (NvBufSurface *) inmap.data;

    AppData* app_data = (AppData*) u_data;

    NvDsObjEncCtxHandle ctx = app_data->ctx_handle;

    /* Iterate each frame metadata in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        /* Iterate object metadata in frame */
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;

            if (!obj_meta)
                continue;

            bool is_suspicious_category = false;

            for (int current_item: SUSPICIOUS_ITEM_IDS) {
                if (current_item == obj_meta->class_id) {
                    is_suspicious_category = true;
                    break;
                }
            }

            if (!is_suspicious_category)
                continue;


            for (DetectedItem& current_detected_item: deteceted_items_in_track) {
                if (obj_meta->object_id == current_detected_item.item_tracking_id) {

                    auto now = std::chrono::system_clock::now();

                    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

                    std::time_t elapsed_seconds = now_c - current_detected_item.now_c;

                    if (elapsed_seconds >= 5) {

                        bool bbox_overlapping = false;

                        for (NvDsMetaList* internal_list_obj = frame_meta->obj_meta_list; internal_list_obj != NULL; internal_list_obj = internal_list_obj->next) {
                            NvDsObjectMeta* internal_obj_meta = (NvDsObjectMeta* ) internal_list_obj->data;

                            if (internal_obj_meta->class_id == 0) {
                                float left = internal_obj_meta->rect_params.left;
                                float top = internal_obj_meta->rect_params.top;
                                float width = internal_obj_meta->rect_params.width;
                                float height = internal_obj_meta->rect_params.height;

                                BBox current_item_bbox {left, top, width, height};

                                if (isOverlap(current_item_bbox, current_detected_item.item_bbox)) {
                                    bbox_overlapping = true;
                                    break;
                                }
                            }
                        }

                        if (bbox_overlapping)
                            continue;


                        obj_meta->text_params.display_text = g_strdup_printf("Suspicious Item");
                        obj_meta->text_params.set_bg_clr = 1;
                        obj_meta->text_params.text_bg_clr.red = 0.0;
                        obj_meta->text_params.text_bg_clr.green = 0.0;
                        obj_meta->text_params.text_bg_clr.blue = 0.0;
                        obj_meta->text_params.text_bg_clr.alpha = 1.0;

                        obj_meta->rect_params.border_color.red = 1.0;
                        obj_meta->rect_params.border_color.green = 0.0;
                        obj_meta->rect_params.border_color.blue = 0.0;
                        obj_meta->rect_params.border_color.alpha = 1.0;

                        auto result = reported_left_items.insert(obj_meta->object_id);

                        if (!result.second)
                            continue;

                        app_data->detected_item_bbox->height = obj_meta->rect_params.height;
                        app_data->detected_item_bbox->width = obj_meta->rect_params.width;
                        app_data->detected_item_bbox->top = obj_meta->rect_params.top;
                        app_data->detected_item_bbox->left = obj_meta->rect_params.left;

                        app_data->detected_item_id = obj_meta->object_id;

                        NvDsObjEncUsrArgs objData = {0};

                        objData.saveImg = true;
                        objData.attachUsrMeta = false;

                        objData.scaleImg = FALSE;
                        objData.scaledWidth = 0;
                        objData.scaledHeight = 0;

                        char fileName[1024] = "suspicious_item.jpg";
                        strcpy(objData.fileNameImg, fileName);

                        objData.objNum = 0;
                        objData.quality = 80;

                        NvDsObjectMeta* specific_obj_meta = new NvDsObjectMeta();
                        specific_obj_meta->rect_params.left = 0;
                        specific_obj_meta->rect_params.top = 0;
                        specific_obj_meta->rect_params.width = ip_surf->surfaceList[0].width;
                        specific_obj_meta->rect_params.height = ip_surf->surfaceList[0].height;

                        // /* Main Function Call */
                        nvds_obj_enc_process((NvDsObjEncCtxHandle) ctx, &objData, ip_surf, specific_obj_meta, frame_meta);
                    }
                }
            }
        }
    }


    return GST_PAD_PROBE_OK;
}

static void decodebin_child_added(GstChildProxy *child_proxy, GObject *object, gchar *name, gpointer user_data) {
    DsSourceBinStruct *bin_struct = (DsSourceBinStruct *)user_data;

    // Print the name of the child element that was added
    g_print("Decodebin child added: %s\n", name);

    // If the child element is another decodebin, connect to its child-added signal
    if (g_strrstr(name, "decodebin") == name) {
        g_signal_connect(G_OBJECT(object), "child-added",
                         G_CALLBACK(decodebin_child_added), user_data);
    }

    // If the child element is a PNG decoder, add a videoconvert element
    if (g_strstr_len(name, -1, "pngdec") == name) {
        bin_struct->vidconv = gst_element_factory_make("videoconvert", "source_vidconv");
        gst_bin_add(GST_BIN(bin_struct->source_bin), bin_struct->vidconv);
    } else {
        bin_struct->vidconv = NULL;
    }
}


static void cb_newpad (GstElement *decodebin, GstPad *decoder_src_pad, gpointer data) {
    g_print("In cb_newpad\n");

    // Retrieve the capabilities of the newly created pad
    GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
    const GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);
    DsSourceBinStruct *bin_struct = (DsSourceBinStruct *)data;
    GstCapsFeatures *features = gst_caps_get_features(caps, 0);

    // Check if the new pad is for video and not audio
    if (!strncmp(name, "video", 5)) {
        // Link the decodebin pad to videoconvert if no hardware decoder is used
        if (bin_struct->vidconv) {
            GstPad *conv_sink_pad = gst_element_get_static_pad(bin_struct->vidconv, "sink");
            if (gst_pad_link(decoder_src_pad, conv_sink_pad)) {
                g_printerr("Failed to link decoderbin src pad to converter sink pad\n");
            }
            g_object_unref(conv_sink_pad);
            if (!gst_element_link(bin_struct->vidconv, bin_struct->nvvidconv)) {
                g_printerr("Failed to link videoconvert to nvvideoconvert\n");
            }
        } else {
            // Link directly to nvvideoconvert if no software conversion is needed
            GstPad *conv_sink_pad = gst_element_get_static_pad(bin_struct->nvvidconv, "sink");
            if (gst_pad_link(decoder_src_pad, conv_sink_pad)) {
                g_printerr("Failed to link decoderbin src pad to converter sink pad\n");
            }
            g_object_unref(conv_sink_pad);
        }

        // Check if the pad features contain NVIDIA memory management
        if (gst_caps_features_contains(features, GST_CAPS_FEATURES_NVMM)) {
            g_print("###Decodebin picked NVIDIA decoder plugin.\n");
        } else {
            g_print("###Decodebin did not pick NVIDIA decoder plugin.\n");
        }
    }
}


static bool create_source_bin(DsSourceBinStruct *ds_source_struct, gchar *uri) {
    gchar bin_name[16] = { };
    GstCaps *caps = NULL;
    GstCapsFeatures *feature = NULL;

    // Initialize elements to NULL
    ds_source_struct->nvvidconv = NULL;
    ds_source_struct->capsfilt = NULL;
    ds_source_struct->source_bin = NULL;
    ds_source_struct->rtspsrc = NULL;

    // Generate a unique name for the source bin
    g_snprintf(bin_name, 15, "source-bin-%02d", ds_source_struct->index);

    // Create a new GstBin for the source
    ds_source_struct->source_bin = gst_bin_new(bin_name);

    // Create elements: rtspsrc, nvvideoconvert, capsfilter
    ds_source_struct->rtspsrc = gst_element_factory_make("rtspsrc", "rtsp-source");
    ds_source_struct->nvvidconv = gst_element_factory_make("nvvideoconvert", "source_nvvidconv");
    ds_source_struct->capsfilt = gst_element_factory_make("capsfilter", "source_capset");

    // Check if all elements were created successfully
    if (!ds_source_struct->source_bin || !ds_source_struct->rtspsrc ||
        !ds_source_struct->nvvidconv || !ds_source_struct->capsfilt) {
        g_printerr("One element in source bin could not be created.\n");
        return false;
    }

    // Set the RTSP URI, username, and password
    g_object_set(G_OBJECT(ds_source_struct->rtspsrc),
                 "location", uri,
                 "user-id", "Admin",
                 "user-pw", "qwerty123",
                 NULL);

    // Connect to the "pad-added" signal of rtspsrc
    g_signal_connect(G_OBJECT(ds_source_struct->rtspsrc), "pad-added",
                     G_CALLBACK(cb_newpad), ds_source_struct);

    // Create and set capabilities for the capsfilter element
    caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", NULL);
    feature = gst_caps_features_new("memory:NVMM", NULL);
    gst_caps_set_features(caps, 0, feature);   
    g_object_set(G_OBJECT(ds_source_struct->capsfilt), "caps", caps, NULL);

    // Add elements to the source bin
    gst_bin_add_many(GST_BIN(ds_source_struct->source_bin),
                     ds_source_struct->rtspsrc, ds_source_struct->nvvidconv,
                     ds_source_struct->capsfilt, NULL);

    // Link rtspsrc to the next element
    if (!gst_element_link(ds_source_struct->nvvidconv, ds_source_struct->capsfilt)) {
        g_printerr("Could not link vidconv and capsfilter\n");
        return false;
    }

    // Create a ghost pad for the source bin
    GstPad *gstpad = gst_element_get_static_pad(ds_source_struct->capsfilt, "src");
    if (!gstpad) {
        g_printerr("Could not find srcpad in '%s'", GST_ELEMENT_NAME(ds_source_struct->capsfilt));
        return false;
    }
    if (!gst_element_add_pad(ds_source_struct->source_bin, gst_ghost_pad_new("src", gstpad))) {
        g_printerr("Could not add ghost pad in '%s'", GST_ELEMENT_NAME(ds_source_struct->capsfilt));
    }
    gst_object_unref(gstpad);

    return true;
}



static bool __create_source_bin(DsSourceBinStruct *ds_source_struct, gchar *uri) {
    gchar bin_name[16] = { };
    GstCaps *caps = NULL;
    GstCapsFeatures *feature = NULL;

    // Initialize elements to NULL
    ds_source_struct->nvvidconv = NULL;
    ds_source_struct->capsfilt = NULL;
    ds_source_struct->source_bin = NULL;
    ds_source_struct->uri_decode_bin = NULL;

    // Generate a unique name for the source bin
    g_snprintf(bin_name, 15, "source-bin-%02d", ds_source_struct->index);

    // Create a new GstBin for the source
    ds_source_struct->source_bin = gst_bin_new(bin_name);

    // Create elements: uridecodebin, nvvideoconvert, capsfilter
    ds_source_struct->uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");
    ds_source_struct->nvvidconv = gst_element_factory_make("nvvideoconvert", "source_nvvidconv");
    ds_source_struct->capsfilt = gst_element_factory_make("capsfilter", "source_capset");

    // Check if all elements were created successfully
    if (!ds_source_struct->source_bin || !ds_source_struct->uri_decode_bin ||
        !ds_source_struct->nvvidconv || !ds_source_struct->capsfilt) {
        g_printerr("One element in source bin could not be created.\n");
        return false;
    }

    // Set the input URI to the uridecodebin element
    g_object_set(G_OBJECT(ds_source_struct->uri_decode_bin), "uri", uri, NULL);

    // Connect to the "pad-added" and "child-added" signals of uridecodebin
    g_signal_connect(G_OBJECT(ds_source_struct->uri_decode_bin), "pad-added",
                     G_CALLBACK(cb_newpad), ds_source_struct);
    g_signal_connect(G_OBJECT(ds_source_struct->uri_decode_bin), "child-added",
                     G_CALLBACK(decodebin_child_added), ds_source_struct);

    // Create and set capabilities for the capsfilter element
    caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12", NULL);
    feature = gst_caps_features_new("memory:NVMM", NULL);
    gst_caps_set_features(caps, 0, feature);
    g_object_set(G_OBJECT(ds_source_struct->capsfilt), "caps", caps, NULL);

    // Add elements to the source bin
    gst_bin_add_many(GST_BIN(ds_source_struct->source_bin),
                     ds_source_struct->uri_decode_bin, ds_source_struct->nvvidconv,
                     ds_source_struct->capsfilt, NULL);

    // Link nvvideoconvert and capsfilter elements
    if (!gst_element_link(ds_source_struct->nvvidconv, ds_source_struct->capsfilt)) {
        g_printerr("Could not link vidconv and capsfilter\n");
        return false;
    }

    // Create a ghost pad for the source bin
    GstPad *gstpad = gst_element_get_static_pad(ds_source_struct->capsfilt, "src");
    if (!gstpad) {
        g_printerr("Could not find srcpad in '%s'", GST_ELEMENT_NAME(ds_source_struct->capsfilt));
        return false;
    }
    if (!gst_element_add_pad(ds_source_struct->source_bin, gst_ghost_pad_new("src", gstpad))) {
        g_printerr("Could not add ghost pad in '%s'", GST_ELEMENT_NAME(ds_source_struct->capsfilt));
    }
    gst_object_unref(gstpad);

    return true;
}

std::vector<std::string> split(std::string str, std::string pattern) {
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            s.erase(0,s.find_first_not_of(" "));
            s.erase(s.find_last_not_of(" ") + 1);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

void die_on_amqp_error(amqp_rpc_reply_t x, const char *context) {
    switch (x.reply_type) {
        case AMQP_RESPONSE_NORMAL:
            return;
        case AMQP_RESPONSE_NONE:
            std::cerr << context << ": missing RPC reply type!" << std::endl;
        break;
        case AMQP_RESPONSE_LIBRARY_EXCEPTION:
            std::cerr << context << ": " << amqp_error_string2(x.library_error) << std::endl;
        break;
        case AMQP_RESPONSE_SERVER_EXCEPTION:
            switch (x.reply.id) {
                case AMQP_CONNECTION_CLOSE_METHOD: {
                    amqp_connection_close_t *m = (amqp_connection_close_t *)x.reply.decoded;
                    std::cerr << context << ": server connection error " << m->reply_code << ", message: " << (char *)m->reply_text.bytes << std::endl;
                    break;
                }
                case AMQP_CHANNEL_CLOSE_METHOD: {
                    amqp_channel_close_t *m = (amqp_channel_close_t *)x.reply.decoded;
                    std::cerr << context << ": server channel error " << m->reply_code << ", message: " << (char *)m->reply_text.bytes << std::endl;
                    break;
                }
                default:
                    std::cerr << context << ": unknown server error, method id " << x.reply.id << std::endl;
                break;
            }
        break;
    }
    exit(1);
}


int main(int argc, char *argv[]) {

    GMainLoop *loop = NULL;
    GstElement
            *streammux = NULL, *sink = NULL,
            *primary_detector = NULL, *tracker = NULL,
            *nvvidconv = NULL, *nvosd = NULL, *nvvidconv1 = NULL,
            *outenc = NULL, *capfilt = NULL, *nvtile = NULL,
            *mux = NULL, *encparse = NULL;
    GstElement *queue1 = NULL, *queue2 = NULL, *queue3 = NULL, *queue4 = NULL,
            *queue5 = NULL, *queue6 = NULL, *queue7 = NULL;

    DsSourceBinStruct source_struct[128];
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;
    GstCaps *caps = NULL;
    GstCapsFeatures *feature = NULL;

    static guint src_cnt = 0;
    guint tiler_rows, tiler_columns;
    perf_measure perf_measure;

    guint gpu_id = 0;
    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";
    ifstream fconfig;
    std::map<string, float> postprocess_params_list;
    bool isYAML = false;
    bool isImage = false;
    bool isStreaming = false;
    GList *g_list = NULL;
    GList *iterator = NULL;
    bool isH264 = true;

    gchar *filepath = NULL;

    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

    // RabbitMQ set up
    amqp_connection_state_t connection = amqp_new_connection();
    amqp_socket_t* socket  = amqp_tcp_socket_new(connection);

    if (!socket) {
        cerr << "Unable to create TCP socket for rabbitmq client" << endl;
        return -1;
    }

    int status = amqp_socket_open(
        socket,
        app_config["message-broker"]["hostname"].as<std::string>().c_str(),
        app_config["message-broker"]["port"].as<int>()
        );
    if (status) {
        cout << "Unable to open TCP socket for rabbitmq client" << endl;
        return -1;
    }

    amqp_login(
        connection, "/", 0, 13072,
        app_config["message-broker"]["amqp-heartbeat"].as<int>(),
        AMQP_SASL_METHOD_PLAIN,
        app_config["message-broker"]["username"].as<std::string>().c_str(),
        app_config["message-broker"]["password"].as<std::string>().c_str()
        );

    amqp_channel_open(connection, 1);
    die_on_amqp_error(amqp_get_rpc_reply(connection), "Opening channel");

    amqp_exchange_declare(
        connection, 1, amqp_cstring_bytes(app_config["message-broker"]["exchange"].as<std::string>().c_str()),
        amqp_cstring_bytes("topic"), 0, 0, 0, 0, amqp_empty_table
        );

    amqp_rpc_reply_t response = amqp_get_rpc_reply(connection);
    die_on_amqp_error(response, "MessageBroker");

    /* Check input arguments */
    if (argc == 2 && (g_str_has_suffix(argv[1], ".yml")
                      || (g_str_has_suffix(argv[1], ".yaml")))) {
        isYAML = TRUE;
        if (nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie") == NVDS_YAML_PARSER_SUCCESS) {
            g_print("pgie_type %d\n", pgie_type);
        }
    }

    gst_init(&argc, &argv);

    signal_catch_setup();

    loop = g_main_loop_new(NULL, FALSE);

    perf_measure.pre_time = GST_CLOCK_TIME_NONE;
    perf_measure.total_time = GST_CLOCK_TIME_NONE;
    perf_measure.count = 0;

    pipeline = gst_pipeline_new("pipeline");

    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr("One or more elements could not be created. Exiting.\n");
        return -1;
    }

    gst_bin_add(GST_BIN(pipeline), streammux);
    g_object_set(G_OBJECT(streammux), "batched-push-timeout", 20000, NULL);

    tracker = gst_element_factory_make("nvtracker", "tracker");

    if (!tracker) {
        g_printerr("Failed to create tracker element\n");
        return -1;
    }

    g_object_set(G_OBJECT(tracker),
        "tracker-width", app_config["tracker"]["tracker-width"].as<int>(),
        "tracker-height",  app_config["tracker"]["tracker-height"].as<int>(),
        "gpu-id", app_config["tracker"]["gpu-id"].as<int>(),
        "ll-lib-file", app_config["tracker"]["ll-lib-file"].as<std::string>().c_str(),
        "ll-config-file", NVDS_TRACKER_CONFIG ,
        NULL
        );

    auto input_sources = app_config["sources"];
    char** source_storage = new char*[input_sources.size()];

    if (!isYAML) {
        for (int index = 0; index < input_sources.size(); index++) {
            source_storage[index] = new char[input_sources[index].size()];
            std::strcpy(source_storage[index], input_sources[index].as<std::string>().c_str());

            g_list = g_list_append(g_list, source_storage[index]);
        }

    } else {
        if (NVDS_YAML_PARSER_SUCCESS != nvds_parse_source_list(&g_list, argv[1], "source-list")) {
            g_printerr("Failed to parse config file\n");
            return -1;
        }
    }

    for (iterator = g_list, src_cnt = 0; iterator; iterator = iterator->next, src_cnt++) {
        /* Source element for reading from the file */
        source_struct[src_cnt].index = src_cnt;

        // Determine if the source is an image
        if (g_strrstr((gchar *) iterator->data, ".jpg") ||
            g_strrstr((gchar *) iterator->data, ".jpeg") ||
            g_strrstr((gchar *) iterator->data, ".png")) {
            isImage = true;
        } else {
            isImage = false;
        }

        // Determine if the source is a streaming source
        if (g_strrstr((gchar *) iterator->data, "rtsp://") ||
            g_strrstr((gchar *) iterator->data, "v4l2://") ||
            g_strrstr((gchar *) iterator->data, "http://") ||
            g_strrstr((gchar *) iterator->data, "rtmp://")) {
            isStreaming = true;
        } else {
            isStreaming = false;
        }

        if (!create_source_bin(&(source_struct[src_cnt]), (gchar *)iterator->data)) {
            g_printerr("Source bin could not be created. Exiting.\n");
            return -1;
        }

        gst_bin_add(GST_BIN(pipeline), source_struct[src_cnt].source_bin);

        g_snprintf(pad_name_sink, 64, "sink_%d", src_cnt);
        sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
        g_print("Request %s pad from streammux\n", pad_name_sink);
        if (!sinkpad) {
            g_printerr("Streammux request sink pad failed. Exiting.\n");
            return -1;
        }

        srcpad = gst_element_get_static_pad(source_struct[src_cnt].source_bin, pad_name_src);
        if (!srcpad) {
            g_printerr("Decoder request src pad failed. Exiting.\n");
            return -1;
        }

        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
            return -1;
        }
        gst_object_unref(sinkpad);
        gst_object_unref(srcpad);
    }

    if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
        primary_detector = gst_element_factory_make("nvinferserver", "primary-infer-engine1");
    } else {
        primary_detector = gst_element_factory_make("nvinfer", "primary-infer-engine1");
    }


    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvid-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvid-converter1");

    capfilt = gst_element_factory_make ("capsfilter", "nvvideo-caps");

    nvtile = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

    queue1 = gst_element_factory_make ("queue", "queue1");
    queue2 = gst_element_factory_make ("queue", "queue2");
    queue3 = gst_element_factory_make ("queue", "queue3");
    queue4 = gst_element_factory_make ("queue", "queue4");
    queue5 = gst_element_factory_make ("queue", "queue5");
    queue6 = gst_element_factory_make ("queue", "queue6");
    queue7 = gst_element_factory_make ("queue", "queue7");

    guint output_type = 0;

    if (isYAML) {
        output_type = ds_parse_group_type(argv[1], "output");
        if (!output_type) {
            g_printerr("No output setting. Exiting.\n");
            return -1;
        }
    } else {
        output_type = 3;
    }

    if (output_type == 1) {
        GString * filename = NULL;
        if (isYAML)
            filename = ds_parse_file_name(argv[1], "output");
        else
            filename = g_string_new(argv[argc-1]);

        if (isImage) {
            outenc = gst_element_factory_make("jpegenc", "jpegenc");
            caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "I420", NULL);
            g_object_set(G_OBJECT(capfilt), "caps", caps, NULL);
            filepath = g_strconcat(filename->str, ".jpg", NULL);
        } else {
            mux = gst_element_factory_make("qtmux", "mp4-mux");
            if (isYAML) {
                isH264 = !(ds_parse_enc_type(argv[1], "output"));
            }

            if (!isH264) {
                encparse = gst_element_factory_make("h265parse", "h265-encparser");
                outenc = gst_element_factory_make("nvv4l2h265enc", "nvvideo-h265enc");
            } else {
                encparse = gst_element_factory_make("h264parse", "h264-encparser");
                outenc = gst_element_factory_make("nvv4l2h264enc", "nvvideo-h264enc");
            }
            filepath = g_strconcat(filename->str, ".mp4", NULL);
            if (isYAML) {
                ds_parse_enc_config(outenc, argv[1], "output");
            } else {
                g_object_set(G_OBJECT(outenc), "bitrate", 4000000, NULL);
            }
            caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "I420", NULL);
            feature = gst_caps_features_new("memory:NVMM", NULL);
            gst_caps_set_features(caps, 0, feature);
            g_object_set(G_OBJECT(capfilt), "caps", caps, NULL);
        }
        sink = gst_element_factory_make("filesink", "nvvideo-renderer");
    } else if (output_type == 2) {
        sink = gst_element_factory_make("fakesink", "fake-renderer");
    } else if (output_type == 3) {
// #ifdef PLATFORM_TEGRA
//         transform = gst_element_factory_make("nvegltransform", "nvegltransform");
//     if (!transform) {
//         g_printerr("nvegltransform element could not be created. Exiting.\n");
//         return -1;
//     }
// #endif
        //sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
        sink = gst_element_factory_make("nvrtspoutsinkbin", "nvvideo-rendererrtsp");
    }

    if (!primary_detector || !nvvidconv || !nvosd || !sink  || !capfilt) {

        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    if (isYAML)
        nvds_parse_streammux(streammux, argv[1], "streammux");
    else
        g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                     MUXER_OUTPUT_HEIGHT, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    if (isStreaming)
        g_object_set(G_OBJECT(streammux), "live-source", true, NULL);

    g_object_set(G_OBJECT(streammux), "batch-size", src_cnt, NULL);

    tiler_rows = (guint) sqrt(src_cnt);
    tiler_columns = (guint) ceil(1.0 * src_cnt / tiler_rows);
    g_object_set(G_OBJECT(nvtile), "rows", tiler_rows, "columns", tiler_columns, "width", 1280, "height", 720, NULL);

    /* Set the config files for the two detectors. The first detector is PGIE which
    * detects the faces. The second detector is SGIE which generates facial landmarks
    * for every face. */
    if (isYAML) {
        nvds_parse_gie(primary_detector, argv[1], "primary-gie");
    } else {
        g_object_set(G_OBJECT(primary_detector), "config-file-path",
                     PRIMARY_ENGINE_CONFIG,
                     "unique-id", PRIMARY_DETECTOR_UID, NULL);
    }


    /* we add a bus message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    gst_bin_add_many (GST_BIN (pipeline), primary_detector, tracker, queue1, queue2, queue3, queue4, queue5,  nvvidconv, nvosd, nvtile, sink, NULL);

    if (!gst_element_link_many (streammux, queue1, primary_detector, tracker,queue2, queue3, nvtile, queue4, nvvidconv, queue5, nvosd, NULL)) {
        g_printerr ("Inferring and tracking elements link failure.\n");
        return -1;
    }

    g_object_set(G_OBJECT(sink), "sync", 0, "async", FALSE, NULL);

    if (output_type == 1) {
        g_object_set(G_OBJECT(sink), "location", filepath, NULL);
        g_object_set(G_OBJECT(sink), "enable-last-sample", false, NULL);

        if (!isImage) {
            gst_bin_add_many(GST_BIN(pipeline), nvvidconv1, outenc, capfilt, queue6, queue7, encparse, mux, NULL);
            if (!gst_element_link_many(nvosd, queue6, nvvidconv1, capfilt, queue7, outenc, encparse, mux, sink, NULL)) {
                g_printerr("OSD and sink elements link failure.\n");
                return -1;
            }
        } else {
            gst_bin_add_many(GST_BIN(pipeline), nvvidconv1, outenc, capfilt, queue6, queue7, NULL);
            if (!gst_element_link_many(nvosd, queue6, nvvidconv1, capfilt, queue7, outenc, sink, NULL)) {
                g_printerr("OSD and sink elements link failure.\n");
                return -1;
            }
        }
        g_free(filepath);
    } else if (output_type == 2) {
        if (!gst_element_link(nvosd, sink)) {
            g_printerr("OSD and sink elements link failure.\n");
            return -1;
        }
    } else if (output_type == 3) {
// #ifdef PLATFORM_TEGRA
//         gst_bin_add_many(GST_BIN(pipeline), transform, queue6, NULL);
//     if (!gst_element_link_many(nvosd, queue6, transform, sink, NULL)) {
//         g_printerr("OSD and sink elements link failure.\n");
//         return -1;
//     }
// #else
//         gst_bin_add(GST_BIN(pipeline), queue6);
//         if (!gst_element_link_many(nvosd, queue6, sink, NULL)) {
//             g_printerr("OSD and sink elements link failure.\n");
//             return -1;
//         }
// #endif

        gst_bin_add(GST_BIN(pipeline), queue6);
        if (!gst_element_link_many(nvosd, queue6, sink, NULL)) {
            g_printerr("OSD and sink elements link failure.\n");
            return -1;
        }
    }


    /* Display the facemarks output on video. Fakesink do not need to display. */
    if(output_type != 2) {
        osd_sink_pad = gst_element_get_static_pad (nvtile, "sink");

        if (!osd_sink_pad)
            g_print ("Unable to get sink pad\n");
        else
           // gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, tile_sink_pad_buffer_probe, NULL, NULL);
        gst_object_unref (osd_sink_pad);
    }

    NvDsObjEncCtxHandle obj_ctx_handle = nvds_obj_enc_create_context (gpu_id);
    if (!obj_ctx_handle) {
        g_print ("Unable to create context\n");
        return -1;
    }

    BBox ctx_detected_item_bbox;

    AppData app_context { &connection, obj_ctx_handle, &ctx_detected_item_bbox, 0 };

    osd_sink_pad = gst_element_get_static_pad (queue2, "src");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_sink_pad_buffer_probe, &perf_measure, NULL);
    gst_object_unref (osd_sink_pad);

    osd_sink_pad = gst_element_get_static_pad (queue3, "src");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, bbox_overlap_verification_probe, &app_context, NULL);
    gst_object_unref (osd_sink_pad);

    osd_sink_pad = gst_element_get_static_pad(queue4, "src");
    if (!osd_sink_pad)
        g_print("Unable to get sink pad\n");
    else
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, message_broker_probe, &app_context, NULL);
    gst_object_unref (osd_sink_pad);


    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);

    if(perf_measure.total_time) {
        g_print ("Average fps %f\n", ((perf_measure.count-1)*src_cnt*1000000.0)/perf_measure.total_time);
    }

    amqp_channel_close(connection, 1, AMQP_REPLY_SUCCESS);
    amqp_connection_close(connection, AMQP_REPLY_SUCCESS);
    amqp_destroy_connection(connection);

    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}

